import streamlit as st
import pandas as pd
import calendar
from ortools.sat.python import cp_model
import io

# --- 初期設定 ---
st.set_page_config(page_title="AIシフトメーカー", layout="wide")
st.title("📅 インタラクティブ・シフトメーカー")

# --- 設定サイドバー ---
with st.sidebar:
    st.header("⚙️ 基本設定")
    year = 2026
    month = st.selectbox("作成月", range(1, 13), index=4)
    
    st.divider()
    st.subheader("⏳ 勤務時間上限（月間）")
    # 個別設定を削除し、共通設定のみに戻しました
    max_h_common = st.number_input("全スタッフ共通の上限(h)", value=177)
    
    st.divider()
    s01_night_limit = st.number_input("スタ01夜勤上限", value=4)
    
    st.divider()
    st.header("📂 データ連携")
    uploaded_file = st.file_uploader("CSVファイルを読み込む", type="csv")

# --- 1. 入力データの準備 ---
emp_names = [f"スタッフ{str(i+1).zfill(2)}" for i in range(12)]
_, num_days = calendar.monthrange(year, month)
day_cols = [f"{d}" for d in range(1, num_days + 1)]

default_df = pd.DataFrame("", index=emp_names, columns=day_cols)

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file, index_col=0)
        input_df = input_df.reindex(index=emp_names, columns=day_cols).fillna("")
        st.sidebar.success("CSVを読み込みました")
    except Exception as e:
        st.sidebar.error(f"読み込みエラー: {e}")
        input_df = default_df
else:
    input_df = default_df

# --- 2. 入力フォーム（データエディタ） ---
st.subheader("📝 勤務希望・確定事項の入力")

edited_df = st.data_editor(
    input_df,
    use_container_width=True,
    column_config={
        col: st.column_config.SelectboxColumn(
            options=["", "日勤", "夜勤入", "夜勤明け", "休み", "会議"],
            width="small"
        ) for col in day_cols
    }
)

csv_buffer = io.StringIO()
edited_df.to_csv(csv_buffer)
st.download_button(
    label="📥 現在の入力内容をCSVで保存する",
    data=csv_buffer.getvalue(),
    file_name=f"shift_input_{year}_{month}.csv",
    mime="text/csv",
)

# --- 3. シフト計算ロジック ---
def solve_shift():
    model = cp_model.CpModel()
    all_days = range(1, num_days + 1)
    all_emps = range(len(emp_names))
    OFF, DAY, N_START, N_END, MEETING = 0, 1, 2, 3, 4
    STATES = [OFF, DAY, N_START, N_END, MEETING]

    shifts = {}
    for e in all_emps:
        for d in all_days:
            for s in STATES:
                shifts[(e, d, s)] = model.NewBoolVar(f'e{e}d{d}s{s}')

    for d in all_days:
        for e in all_emps:
            model.Add(sum(shifts[(e, d, s)] for s in STATES) == 1)
        
        # 人数制限（会議は含めない）
        model.Add(sum(shifts[(e, d, DAY)] for e in all_emps) >= 3)
        model.Add(sum(shifts[(e, d, DAY)] for e in all_emps) <= 5)
        model.Add(sum(shifts[(e, d, N_START)] for e in all_emps) == 2)
        model.Add(sum(shifts[(e, d, N_END)] for e in all_emps) == 2)

    for e_idx, e_name in enumerate(emp_names):
        for d_idx in range(num_days):
            val = edited_df.iloc[e_idx, d_idx]
            day_num = d_idx + 1
            if val == "日勤": model.Add(shifts[(e_idx, day_num, DAY)] == 1)
            if val == "夜勤入": model.Add(shifts[(e_idx, day_num, N_START)] == 1)
            if val == "夜勤明け": model.Add(shifts[(e_idx, day_num, N_END)] == 1)
            if val == "休み": model.Add(shifts[(e_idx, day_num, OFF)] == 1)
            if val == "会議": model.Add(shifts[(e_idx, day_num, MEETING)] == 1)

    for e in all_emps:
        for d in range(1, num_days):
            model.Add(shifts[(e, d, N_START)] == shifts[(e, d+1, N_END)])
        for d in range(1, num_days - 4):
            model.Add(sum(shifts[(e, d + i, OFF)] for i in range(6)) >= 1)
        
        # --- 時間計算（共通上限を適用） ---
        hrs = [
            shifts[(e, d, DAY)] * 8 + 
            shifts[(e, d, N_START)] * 6 + 
            shifts[(e, d, N_END)] * 8 + 
            shifts[(e, d, MEETING)] * 8 
            for d in all_days
        ]
        
        # 全員一律で共通設定を参照
        model.Add(sum(hrs) <= max_h_common)
    
    model.Add(sum(shifts[(0, d, N_START)] for d in all_days) == s01_night_limit)
    for e in range(7, 12):
        for d in all_days:
            model.Add(shifts[(e, d, N_START)] == 0)
            model.Add(shifts[(e, d, N_END)] == 0)

    obj_terms = []
    for e in all_emps:
        for d in range(1, num_days+1):
            obj_terms.append(shifts[(e, d, DAY)] * 5)

    model.Maximize(sum(obj_terms))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 180.0
    solver.parameters.num_search_workers = 2
    status = solver.Solve(model)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        res_data = []
        name_map = {DAY: "日勤", N_START: "夜勤入", N_END: "夜勤明", OFF: "休み", MEETING: "会議"}
        for e in all_emps:
            row = {"スタッフ": emp_names[e]}
            for d in all_days:
                for s, name in name_map.items():
                    if solver.Value(shifts[(e, d, s)]): row[f"{d}"] = name
            res_data.append(row)
        return pd.DataFrame(res_data).set_index("スタッフ")
    return None

# --- 4. 出力表示 ---
if st.button("🚀 シフトを作成する"):
    with st.spinner("AIが計算中..."):
        result_df = solve_shift()
        if result_df is not None:
            st.success("シフトが完成しました！")
            st.dataframe(result_df, use_container_width=True)
            
            st.subheader("☀️ 日毎の日勤人数")
            daily_day_counts = (result_df == "日勤").sum()
            daily_summary = pd.DataFrame(daily_day_counts).T
            daily_summary.index = ["日勤合計"]
            st.dataframe(daily_summary, use_container_width=True)

            res_csv = io.StringIO()
            result_df.to_csv(res_csv)
            st.download_button(
                label="📊 完成したシフトをCSVで書き出す",
                data=res_csv.getvalue(),
                file_name=f"shift_result_{year}_{month}.csv",
                mime="text/csv",
            )
            
            st.subheader("📊 最終集計（個人別）")
            counts = result_df.T.apply(pd.Series.value_counts).fillna(0).astype(int)

            # 勤務時間合計の計算
            hours_map = {"日勤": 8, "夜勤入": 6, "夜勤明": 8, "休み": 0, "会議": 8}
            
            def calculate_hours(series):
                return sum(series.map(lambda x: hours_map.get(x, 0)))
            
            total_hours = result_df.apply(calculate_hours, axis=1)
            counts.loc["勤務時間合計"] = total_hours
            
            st.table(counts)
        else:
            st.error("解が見つかりませんでした。")
