import streamlit as st
import pandas as pd
import calendar
from ortools.sat.python import cp_model

# --- 初期設定 ---
st.set_page_config(page_title="AIシフトメーカー", layout="wide")
st.title("📅 インタラクティブ・シフトメーカー")

# --- 設定サイドバー ---
with st.sidebar:
    st.header("⚙️ 基本設定")
    year = 2026
    month = st.selectbox("作成月", range(1, 13), index=3)
    max_h = st.number_input("上限時間(01-08)", value=177)
    s01_night_limit = st.number_input("スタ01夜勤上限", value=4)
    st.info("※1日の夜勤明けも下の表で直接選択してください。")

# --- 1. 入力フォーム（データエディタ）の作成 ---
st.subheader("📝 勤務希望・確定事項の入力")
st.markdown("""
- **空欄**: AIが自動で埋めます
- **各項目**: その勤務で固定します
- **注意**: 「夜勤入」を入れた翌日は、自動的に「夜勤明け」になります。前月末からの引き継ぎは1日に「夜勤明け」を入力してください。
""")

emp_names = [f"スタッフ{str(i+1).zfill(2)}" for i in range(12)]
_, num_days = calendar.monthrange(year, month)
days = [f"{d}({['月','火','水','木','金','土','日'][calendar.weekday(year, month, d)]})" for d in range(1, num_days + 1)]

# 初期データの作成（すべて空欄）
init_df = pd.DataFrame("", index=days, columns=emp_names)

# データエディタを表示
edited_df = st.data_editor(
    init_df,
    use_container_width=True,
    column_config={
        col: st.column_config.SelectboxColumn(
            options=["", "日勤", "夜勤入", "夜勤明け", "休み"],
            width="small"
        ) for col in emp_names
    }
)

# --- 2. シフト計算ロジック ---
def solve_shift():
    model = cp_model.CpModel()
    all_days = range(1, num_days + 1)
    all_emps = range(len(emp_names))
    OFF, DAY, N_START, N_END = 0, 1, 2, 3
    STATES = [OFF, DAY, N_START, N_END]

    shifts = {}
    for e in all_emps:
        for d in all_days:
            for s in STATES:
                shifts[(e, d, s)] = model.NewBoolVar(f'e{e}d{d}s{s}')

    # --- 基本制約（人数固定） ---
    for d in all_days:
        for e in all_emps:
            model.Add(sum(shifts[(e, d, s)] for s in STATES) == 1)
        model.Add(sum(shifts[(e, d, DAY)] for e in all_emps) == 4)
        model.Add(sum(shifts[(e, d, N_START)] for e in all_emps) == 2)
        model.Add(sum(shifts[(e, d, N_END)] for e in all_emps) == 2)

    # --- UIでの入力内容をハード制約として追加 ---
    for e_idx, e_name in enumerate(emp_names):
        for d_idx, d_label in enumerate(days):
            val = edited_df.iloc[d_idx, e_idx]
            day_num = d_idx + 1
            if val == "日勤": model.Add(shifts[(e_idx, day_num, DAY)] == 1)
            if val == "夜勤入": model.Add(shifts[(e_idx, day_num, N_START)] == 1)
            if val == "夜勤明け": model.Add(shifts[(e_idx, day_num, N_END)] == 1)
            if val == "休み": model.Add(shifts[(e_idx, day_num, OFF)] == 1)

    # --- 共通ルール ---
    for e in all_emps:
        for d in range(1, num_days):
            # 夜勤入の翌日は夜勤明け（自動連動）
            model.Add(shifts[(e, d, N_START)] == shifts[(e, d+1, N_END)])
            # 夜勤明けの翌日は日勤禁止
            model.Add(shifts[(e, d, N_END)] + shifts[(e, d+1, DAY)] <= 1)
        
        # 6連勤制限
        for d in range(1, num_days - 5):
            model.Add(sum(shifts[(e, d + i, OFF)] for i in range(7)) >= 1)

    # スタッフ別制限
    for e in range(8): # 01-08 時間制限
        hrs = [shifts[(e, d, DAY)]*8 + shifts[(e, d, N_START)]*6 + shifts[(e, d, N_END)]*8 for d in all_days]
        model.Add(sum(hrs) <= max_h)
    
    model.Add(sum(shifts[(0, d, N_START)] for d in all_days) <= s01_night_limit) # スタ01夜勤
    
    for e in range(7, 12): # 08-12 夜勤不可
        for d in all_days:
            model.Add(shifts[(e, d, N_START)] == 0)
            model.Add(shifts[(e, d, N_END)] == 0)
            
    # スタッフ12は8日勤務固定
    s12_work = [shifts[(11, d, s)] for d in all_days for s in [DAY, N_START, N_END]]
    model.Add(sum(s12_work) == 8)

    # --- 目的関数 ---
    obj_terms = []
    # 基本の出勤確保
    for e in range(11):
        for d in all_days:
            obj_terms.append(shifts[(e, d, DAY)] * 10)
            if e < 7: obj_terms.append(shifts[(e, d, N_START)] * 15)

    # 休みを分散させる（4日間隔にボーナス）
    for e in all_emps:
        for d in range(1, num_days - 4):
            is_sep = model.NewBoolVar(f'sep_e{e}d{d}')
            v1, v2 = shifts[(e, d, OFF)], shifts[(e, d+4, OFF)]
            model.Add(is_sep <= v1)
            model.Add(is_sep <= v2)
            model.Add(is_sep >= v1 + v2 - 1)
            obj_terms.append(is_sep * 30)

    model.Maximize(sum(obj_terms))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 180.0 # 3分間じっくり計算
    status = solver.Solve(model)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        res = []
        for d in all_days:
            row = {"日付": days[d-1]}
            for e in all_emps:
                for s, name in {DAY:"日勤", N_START:"夜勤入", N_END:"夜勤明", OFF:"休み"}.items():
                    if solver.Value(shifts[(e, d, s)]): row[emp_names[e]] = name
            res.append(row)
        return pd.DataFrame(res).set_index("日付")
    return None

if st.button("🚀 シフトを作成する"):
    with st.spinner("AIが最適な組み合わせを計算中... (最大3分)"):
        result_df = solve_shift()
        if result_df is not None:
            st.success("シフトが完成しました！")
            st.dataframe(result_df, use_container_width=True)
            # 集計表示
            st.subheader("📊 最終集計")
            counts = result_df.apply(pd.Series.value_counts).fillna(0).astype(int)
            st.table(counts)
        else:
            st.error("解が見つかりませんでした。入力した「夜勤入」と「夜勤明け」の整合性（翌日になっているか等）や、人数制限を確認してください。")
