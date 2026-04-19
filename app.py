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
    month = st.selectbox("作成月", range(1, 13), index=3)  # デフォルト4月
    max_h = st.number_input("上限時間(全スタッフ共通)", value=177)
    s01_night_limit = st.number_input("スタ01夜勤上限", value=4)
    st.info("※最低15日出勤は「目標」です。無理な場合は15日を下回る解を出力します。")

# --- 1. 入力フォーム（データエディタ）の作成 ---
st.subheader("📝 勤務希望・確定事項の入力")
st.markdown("""
- **スタッフ（行）× 日付（列）**の表です。
- 確定している勤務を選択してください。空欄はAIが自動で埋めます。
- **1日の夜勤明け**もこの表で指定してください。
""")

emp_names = [f"スタッフ{str(i+1).zfill(2)}" for i in range(12)]
_, num_days = calendar.monthrange(year, month)
day_cols = [f"{d}" for d in range(1, num_days + 1)]

# 初期データの保持（セッション状態を使用）
if "df_input" not in st.session_state or st.session_state.get('last_month') != month:
    st.session_state.df_input = pd.DataFrame("", index=emp_names, columns=day_cols)
    st.session_state.last_month = month

# データエディタを表示
edited_df = st.data_editor(
    st.session_state.df_input,
    use_container_width=True,
    column_config={
        col: st.column_config.SelectboxColumn(
            options=["", "日勤", "夜勤入", "夜勤明け", "休み"],
            width="small"
        ) for col in day_cols
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

    # --- 基本制約（ハード：人数固定） ---
    for d in all_days:
        for e in all_emps:
            model.Add(sum(shifts[(e, d, s)] for s in STATES) == 1)
        model.Add(sum(shifts[(e, d, DAY)] for e in all_emps) == 4)
        model.Add(sum(shifts[(e, d, N_START)] for e in all_emps) == 2)
        model.Add(sum(shifts[(e, d, N_END)] for e in all_emps) == 2)

    # --- UIからの入力内容を反映（ハード） ---
    for e_idx, e_name in enumerate(emp_names):
        for d_idx in range(num_days):
            val = edited_df.iloc[e_idx, d_idx]
            day_num = d_idx + 1
            if val == "日勤": model.Add(shifts[(e_idx, day_num, DAY)] == 1)
            if val == "夜勤入": model.Add(shifts[(e_idx, day_num, N_START)] == 1)
            if val == "夜勤明け": model.Add(shifts[(e_idx, day_num, N_END)] == 1)
            if val == "休み": model.Add(shifts[(e_idx, day_num, OFF)] == 1)

    # --- 共通ルール（ハード） ---
    for e in all_emps:
        for d in range(1, num_days):
            # 夜勤入の翌日は夜勤明け
            model.Add(shifts[(e, d, N_START)] == shifts[(e, d+1, N_END)])
            # 夜勤明けの翌日は日勤禁止
            model.Add(shifts[(e, d, N_END)] + shifts[(e, d+1, DAY)] <= 1)
        # 6連勤制限
        for d in range(1, num_days - 5):
            model.Add(sum(shifts[(e, d + i, OFF)] for i in range(7)) >= 1)
        # 時間制限
        hrs = [shifts[(e, d, DAY)]*8 + shifts[(e, d, N_START)]*6 + shifts[(e, d, N_END)]*8 for d in all_days]
        model.Add(sum(hrs) <= max_h)
    
    # スタッフ01夜勤上限
    model.Add(sum(shifts[(0, d, N_START)] for d in all_days) <= s01_night_limit)
    
    # 夜勤不可（スタッフ08-12）
    for e in range(7, 12):
        for d in all_days:
            model.Add(shifts[(e, d, N_START)] == 0)
            model.Add(shifts[(e, d, N_END)] == 0)

    # --- 目的関数（ソフト制約） ---
    obj_terms = []

    # 1. 出勤日数の確保（ソフト制約：15日目標）
    for e in all_emps:
        work_vars = [shifts[(e, d, s)] for d in all_days for s in [DAY, N_START, N_END]]
        actual_work_days = sum(work_vars)
        under_15 = model.NewIntVar(0, 15, f'under15_e{e}')
        model.AddMinEquality(under_15, [actual_work_days, 15])
        obj_terms.append(under_15 * 100) # 15日までは高配点
        obj_terms.append(actual_work_days * 10) # 全体の底上げ

    # 2. 休みを等間隔にするボーナス
    for e in all_emps:
        for d in range(1, num_days - 4):
            is_sep = model.NewBoolVar(f'sep_e{e}d{d}')
            v1, v2 = shifts[(e, d, OFF)], shifts[(e, d+4, OFF)]
            model.Add(is_sep <= v1)
            model.Add(is_sep <= v2)
            model.Add(is_sep >= v1 + v2 - 1)
            obj_terms.append(is_sep * 30)

    # 3. 夜勤の積極割り振り
    for e in range(7):
        for d in all_days:
            obj_terms.append(shifts[(e, d, N_START)] * 20)

    model.Maximize(sum(obj_terms))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 180.0
    solver.parameters.num_search_workers = 2
    status = solver.Solve(model)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        res_data = []
        for e in all_emps:
            row = {"スタッフ": emp_names[e]}
            for d in all_days:
                for s, name in {DAY:"日勤", N_START:"夜勤入", N_END:"夜勤明", OFF:"休み"}.items():
                    if solver.Value(shifts[(e, d, s)]): row[f"{d}"] = name
            res_data.append(row)
        return pd.DataFrame(res_data).set_index("スタッフ")
    return None

# --- 3. 結果の表示と装飾 ---
if st.button("🚀 シフトを作成する"):
    with st.spinner("AIが最適な組み合わせを計算中... (最大3分)"):
        result_df = solve_shift()
        if result_df is not None:
            st.success("シフトが完成しました！")

            # 背景色のスタイル関数（縞々模様）
            def style_zebra(df):
                color_even_row = 'background-color: #f2f2f2' # 偶数行：薄グレー
                color_even_col = 'background-color: #f0f7ff' # 偶数列：薄青
                color_both = 'background-color: #e5f1ff'     # 重なり
                
                res = pd.DataFrame('', index=df.index, columns=df.columns)
                for r in range(len(df)):
                    for c in range(len(df.columns)):
                        is_even_row = (r % 2 == 1)
                        is_even_col = (c % 2 == 1)
                        if is_even_row and is_even_col: res.iloc[r, c] = color_both
                        elif is_even_row: res.iloc[r, c] = color_even_row
                        elif is_even_col: res.iloc[r, c] = color_even_col
                return res

            st.dataframe(
                result_df.style.apply(style_zebra, axis=None),
                use_container_width=True
            )
            
            # 集計
            st.subheader("📊 最終集計")
            counts = result_df.T.apply(pd.Series.value_counts).fillna(0).astype(int)
            st.table(counts)
        else:
            st.error("解が見つかりませんでした。入力した「夜勤入」の翌日に「休み」を入れるなど、矛盾がないか確認してください。")
