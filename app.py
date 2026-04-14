import streamlit as st
import pandas as pd
import calendar
import json
import google.generativeai as genai
from ortools.sat.python import cp_model

# --- 手順1: SecretsからAPIキーを取得 ---
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    model = genai.GenerativeModel(
        model_name='models/gemini-2.0-flash', # 最新モデルに更新
        generation_config={"response_mime_type": "application/json"}
    )
else:
    st.error("APIキーが設定されていません。Streamlit CloudのSettingsから設定してください。")
    st.stop()

def get_constraints(user_input, employees):
    mapping_str = "\n".join([f"ID {i}: {name}" for i, name in enumerate(employees)])
    prompt = f"スタッフの休み希望を抽出しJSONで出力せよ。\n対応表:\n{mapping_str}\n入力:「{user_input}」"
    response = model.generate_content(prompt)
    return json.loads(response.text)

def create_shift(year, month, requests_data, max_hours, s01_night_limit):
    _, num_days = calendar.monthrange(year, month)
    all_days = range(1, num_days + 1)
    employees = [f"スタッフ{str(i+1).zfill(2)}" for i in range(12)]
    all_employees = range(len(employees))
    
    OFF, DAY, N_START, N_END = 0, 1, 2, 3
    STATES = [OFF, DAY, N_START, N_END]

    model_ortools = cp_model.CpModel()
    shifts = {}
    for e in all_employees:
        for d in all_days:
            for s in STATES:
                shifts[(e, d, s)] = model_ortools.NewBoolVar(f'e{e}d{d}s{s}')

    # 基本制約（人数固定）
    for d in all_days:
        for e in all_employees:
            model_ortools.Add(sum(shifts[(e, d, s)] for s in STATES) == 1)
        model_ortools.Add(sum(shifts[(e, d, DAY)] for e in all_employees) == 4)
        model_ortools.Add(sum(shifts[(e, d, N_START)] for e in all_employees) == 2)
        model_ortools.Add(sum(shifts[(e, d, N_END)] for e in all_employees) == 2)

    # 1. 勤務時間の制約 (スタッフ01〜08)
    for e in range(8): 
        total_hours = []
        for d in all_days:
            total_hours.append(shifts[(e, d, DAY)] * 8)
            total_hours.append(shifts[(e, d, N_START)] * 6)
            total_hours.append(shifts[(e, d, N_END)] * 8)
        model_ortools.Add(sum(total_hours) <= max_hours)

    # 2. 連勤制限（最大6連勤まで）
    # 7日間の中に必ず1日は休みが必要
    for e in all_employees:
        for d in range(1, num_days - 5):
            # d日からd+6日までの7日間で、休み(OFF)の合計が1以上
            model_ortools.Add(sum(shifts[(e, d + i, OFF)] for i in range(7)) >= 1)

    # 3. 最低出勤日数の確保 (スタッフ01〜11は15日以上)
    for e in range(11): # 01-11
        work_days = []
        for d in all_days:
            work_days.append(shifts[(e, d, DAY)])
            work_days.append(shifts[(e, d, N_START)])
            work_days.append(shifts[(e, d, N_END)])
        model_ortools.Add(sum(work_days) >= 15)

    # 4. 夜勤の個別制約
    s01_night_counts = [shifts[(0, d, N_START)] for d in all_days]
    model_ortools.Add(sum(s01_night_counts) <= s01_night_limit)
    for e in range(7, 12): # スタッフ08-12は夜勤不可
        for d in all_days:
            model_ortools.Add(shifts[(e, d, N_START)] == 0)
            model_ortools.Add(shifts[(e, d, N_END)] == 0)

    # 5. スタッフ12の勤務日数固定
    s12_work = [shifts[(11, d, s)] for d in all_days for s in [DAY, N_START, N_END]]
    model_ortools.Add(sum(s12_work) == 8)

    # 6. 夜勤明けの翌日は日勤禁止
    for e in all_employees:
        for d in range(1, num_days):
            model_ortools.Add(shifts[(e, d, N_START)] == shifts[(e, d+1, N_END)])
            model_ortools.Add(shifts[(e, d, N_END)] + shifts[(e, d+1, DAY)] <= 1)

    # --- 目的関数の最適化 ---
    obj_terms = []
    # 休み希望の反映
    reqs = requests_data.get("requests", {}) if isinstance(requests_data, dict) else {}
    for e_id_str, dates in reqs.items():
        try:
            e_id = int(e_id_str)
            for d in dates:
                if 1 <= d <= num_days:
                    obj_terms.append(shifts[(e_id, d, OFF)] * 100)
        except: continue

    # スタッフ01〜07の稼働を優先
    for e in range(7):
        for d in all_days:
            obj_terms.append(shifts[(e, d, DAY)] * 30)
            obj_terms.append(shifts[(e, d, N_START)] * 30)

    model_ortools.Maximize(sum(obj_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30.0
    status = solver.Solve(model_ortools)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        res = []
        for d in all_days:
            w = ["月","火","水","木","金","土","日"][calendar.weekday(year, month, d)]
            row = {"日付": f"{d}({w})"}
            for e in all_employees:
                for s, name in {DAY:"日勤", N_START:"夜勤入", N_END:"夜勤明", OFF:"休み"}.items():
                    if solver.Value(shifts[(e, d, s)]): row[employees[e]] = name
            res.append(row)
        return pd.DataFrame(res).set_index("日付")
    return None

# --- UI部分 ---
st.title("📅 AIシフトメーカー (本格運用版)")

with st.expander("⚙️ 詳細設定", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1: target_month = st.selectbox("作成月", range(1, 13), index=3)
    with col2: max_h = st.number_input("上限時間(01-08)", value=177)
    with col3: s01_night = st.number_input("スタ01夜勤上限", value=4, min_value=0)

user_query = st.text_area("休み希望を入力してください", placeholder="スタッフ01は10日休み、など")

if st.button("✨ シフトを作成"):
    emp_names = [f"スタッフ{str(i+1).zfill(2)}" for i in range(12)]
    with st.spinner("連勤と最低日数を調整しながら計算中..."):
        try:
            data = get_constraints(user_query, emp_names)
            df = create_shift(2026, target_month, data, max_h, s01_night)
            if df is not None:
                st.success("完成しました！")
                st.dataframe(df, use_container_width=True)
                
                # 集計表示
                st.subheader("📊 今月の出勤日数チェック")
                work_counts = df.apply(lambda x: x[x != "休み"].count())
                st.table(pd.DataFrame(work_counts, columns=["出勤日数"]))
            else:
                st.error("解が見つかりませんでした。条件を少し緩めて試してください。")
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
