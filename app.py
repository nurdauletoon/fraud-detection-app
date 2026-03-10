import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- ВАЖНО: Настройка Matplotlib для работы без GUI ---
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
)
import shap
import os
import requests
from streamlit_lottie import st_lottie
import time

# --- ИМПОРТЫ ДЛЯ МОЩНЫХ МОДЕЛЕЙ ---
try:
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    from sklearn.ensemble import RandomForestClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
except ImportError:
    pass 

# ==========================================
# 🎨 КОНФИГУРАЦИЯ СТРАНИЦЫ
# ==========================================
st.set_page_config(
    page_title="AI Fraud Guard Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 🛠️ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================

# 1. Загрузка анимаций (Lottie)
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# 2. Загрузка модели и колонок
@st.cache_resource
def load_model_system():
    """Загружает модель и список колонок с диска"""
    if os.path.exists('model.pkl') and os.path.exists('model_columns.pkl'):
        try:
            model = joblib.load('model.pkl')
            cols = joblib.load('model_columns.pkl')
            return model, cols
        except Exception as e:
            st.error(f"Критическая ошибка загрузки модели: {e}")
            st.warning("Убедитесь, что установлены библиотеки: xgboost, lightgbm, catboost, scikit-learn")
            return None, None
    return None, None

# 3. Умная загрузка CSV (СУПЕР-РОБАСТНАЯ)
@st.cache_data
def load_data(uploaded_file):
    """
    Автоматически определяет:
    1. Кодировку (utf-8, cp1251)
    2. Разделитель (; или ,)
    3. Заголовок (1-я или 2-я строка)
    """
    if uploaded_file is None:
        return None
    
    encodings = ['utf-8', 'cp1251', 'windows-1251']
    separators = [',', ';']
    headers = [0, 1] # Пробуем читать с 1-й строки и со 2-й (если есть мусор в начале)
    
    for enc in encodings:
        for sep in separators:
            for header_row in headers:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=enc, sep=sep, header=header_row)
                    
                    # Проверка успешности: если нашлись ключевые колонки
                    # Ищем 'Class' или 'amount' (в любом регистре)
                    cols_lower = [c.lower() for c in df.columns]
                    if 'class' in cols_lower or 'amount' in cols_lower or 'v1' in cols_lower:
                        
                        # --- АВТО-КОРРЕКЦИЯ КОЛОНОК ---
                        # Переименовываем 'amount' -> 'Amount', 'class' -> 'Class'
                        rename_map = {}
                        for col in df.columns:
                            if col.lower() == 'amount': rename_map[col] = 'Amount'
                            if col.lower() == 'class': rename_map[col] = 'Class'
                        
                        if rename_map:
                            df = df.rename(columns=rename_map)
                            
                        # --- АВТО-ОЧИСТКА ЧИСЕЛ (150,00 -> 150.00) ---
                        for col in df.columns:
                            if df[col].dtype == 'object':
                                try:
                                    # Пробуем превратить в число, меняя запятую на точку
                                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.', regex=False))
                                except:
                                    pass
                                    
                        return df
                except:
                    continue
            
    st.error("❌ Не удалось распознать формат файла. Проверьте, что это CSV.")
    return None

# 4. Парсинг строки из поиска
def smart_parse_input(text_input, model_cols):
    """
    Интеллектуальный парсер:
    1. Поддерживает формат: V1=0.5, V2=-1.2, Amount=200
    2. Поддерживает просто список чисел
    3. Сам определяет порядок
    """
    try:
        text_input = text_input.strip()

        # --- 1. Если формат с названиями колонок ---
        if "=" in text_input:
            data_dict = {}
            pairs = text_input.replace("\n", ",").split(",")

            for pair in pairs:
                if "=" in pair:
                    key, value = pair.split("=")
                    key = key.strip()
                    value = float(value.strip())
                    data_dict[key] = value

            # Создаем строку с нулями
            row = {col: 0.0 for col in model_cols}

            # Заполняем совпадающие колонки
            for key in data_dict:
                if key in row:
                    row[key] = data_dict[key]

            return pd.DataFrame([row])

        # --- 2. Если просто числа ---
        else:
            clean_text = text_input.replace('\n', ',').replace(';', ',')
            values = [float(x.strip()) for x in clean_text.split(',') if x.strip()]

            if len(values) == len(model_cols):
                return pd.DataFrame([values], columns=model_cols)

            elif len(values) < len(model_cols):
                st.warning(f"Получено {len(values)} значений, ожидается {len(model_cols)}. Недостающие заполнены нулями.")
                values += [0.0] * (len(model_cols) - len(values))
                return pd.DataFrame([values], columns=model_cols)

            else:
                st.warning("Лишние значения будут обрезаны.")
                values = values[:len(model_cols)]
                return pd.DataFrame([values], columns=model_cols)

    except Exception as e:
        st.error(f"Ошибка обработки данных: {e}")
        return None

# ==========================================
# 🖌️ ЗАГРУЗКА АСЕТОВ И СТИЛЕЙ
# ==========================================

lottie_security = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_V9t630.json") 
lottie_scanning = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_tij5n82q.json") 
lottie_robot = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_mvmhhrvo.json") 

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; font-family: 'Helvetica Neue', sans-serif; }
    h1, h2, h3 { color: #2c3e50; font-weight: 700; }
    div[data-testid="metric-container"] {
        background-color: #ffffff; border: 1px solid #e0e0e0;
        padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .stButton>button { border-radius: 20px; font-weight: bold; }
    .stTextInput>div>div>input { border-radius: 25px; border: 2px solid #3498db; padding: 10px 20px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 🖥️ БОКОВАЯ ПАНЕЛЬ (SIDEBAR)
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2092/2092663.png", width=80)
    st.title("Fraud Guard AI")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("📂 Загрузить файл данных", type=['csv'])
    
    st.subheader("⚙️ Чувствительность")
    threshold = st.slider("Порог (Threshold)", 0.0, 1.0, 0.4, 0.01, 
                          help="Чем выше порог, тем меньше тревог, но выше шанс пропустить мошенника.")
    
    st.info(f"**Текущий порог:** {threshold}")
    st.markdown("---")
    st.caption("Powered by Stacking Ensemble (XGB+LGBM+Cat+RF)")

# ==========================================
# 🏠 ГЛАВНЫЙ ЭКРАН
# ==========================================

col1, col2 = st.columns([3, 1])
with col1:
    st.title("🛡️ Система Детекции Мошенничества")
    st.markdown("#### Enterprise AI Security System")
with col2:
    if lottie_security:
        st_lottie(lottie_security, height=100, key="sec_anim")

# Загрузка системы
model, model_columns = load_model_system()

if model is None:
    st.error("❌ **Система не готова!** Файлы `model.pkl` и `model_columns.pkl` отсутствуют.")
    st.stop()

# --- ПОИСКОВАЯ СТРОКА (SMART SEARCH) ---
st.markdown("### ⚡ Быстрая проверка")
search_query = st.text_input("", placeholder="Вставьте данные транзакции (например: 0.1, -1.5, 200.0...) и нажмите Enter")

if search_query:
    with st.spinner("🤖 AI анализирует транзакцию..."):
        time.sleep(0.3) # Имитация работы
        input_df = smart_parse_input(search_query, model_columns)
        
        if input_df is not None:
            # Предсказание
            prob = model.predict_proba(input_df)[0, 1]
            pred = 1 if prob >= threshold else 0

# --- Risk Level логикасы ---
            if prob < 0.30:
             risk_level = "🟢 LOW RISK"
             risk_color = "normal"
            elif prob < 0.70:
              risk_level = "🟡 MEDIUM RISK"
              risk_color = "off"
            else:
              risk_level = "🔴 HIGH RISK"
              risk_color = "inverse"
            pred = 1 if prob >= threshold else 0
            
            # Вывод результатов
            res_col1, res_col2, res_col3 = st.columns([1, 2, 1])
            
            with res_col2:
                if pred == 1:
                    st.error(f"🚨 **ОБНАРУЖЕНО МОШЕННИЧЕСТВО!**")
                    st.metric("Вероятность риска", f"{prob*100:.2f}%", delta=f"+{(prob-threshold)*100:.1f}%", delta_color="inverse")
                    if lottie_scanning:
                        st_lottie(lottie_scanning, height=150, key="scan_fraud")
                else:
                    st.success(f"✅ **Транзакция чиста**")
                    st.metric("Вероятность риска", f"{prob*100:.2f}%", delta=f"{(prob-threshold)*100:.1f}%", delta_color="normal")
            
            with st.expander("🔍 Посмотреть обработанные данные"):
                st.dataframe(input_df)
        else:
            st.warning("⚠️ Формат данных не распознан. Используйте числа, разделенные запятой.")

st.markdown("---")

# ==========================================
# 📊 ВНУТРЕННИЕ ВКЛАДКИ
# ==========================================
tab_eda, tab_batch, tab_ai = st.tabs(["📊 Аналитика (EDA)", "📂 Массовая Проверка", "🧠 AI Объяснения (SHAP)"])

# --- TAB 1: EDA ---
with tab_eda:
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            st.success(f"Файл успешно загружен: {df.shape[0]} строк, {df.shape[1]} колонок")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🥧 Баланс классов")
                if 'Class' in df.columns:
                    fig = px.pie(df, names='Class', title='Мошенники (1) vs Нормальные (0)', 
                                 color_discrete_sequence=['#2ecc71', '#e74c3c'], hole=0.4)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("ℹ️ Колонка 'Class' не найдена (показана общая статистика).")
                    
            with col2:
                st.subheader("💰 Распределение сумм")
                if 'Amount' in df.columns:
                    # Фильтруем топ 1% выбросов для красивого графика
                    try:
                        filtered_df = df[df['Amount'] < df['Amount'].quantile(0.99)]
                        fig = px.histogram(filtered_df, x="Amount", nbins=50, title="Гистограмма сумм (99% квантиль)", 
                                           color_discrete_sequence=['#3498db'])
                        st.plotly_chart(fig, use_container_width=True)
                    except:
                        st.warning("Не удалось построить график сумм (проверьте формат данных).")
                else:
                    st.info("ℹ️ Колонка 'Amount' не найдена.")
            
            st.subheader("🔥 Корреляционная матрица (Топ-15 признаков)")
            with st.expander("Показать карту корреляций", expanded=True):
                # Берем только числовые колонки
                num_cols = df.select_dtypes(include=np.number).columns[:15]
                if len(num_cols) > 1:
                    corr = df[num_cols].corr()
                    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Недостаточно числовых данных для корреляции.")
                    
    else:
        st.info("👆 Загрузите файл в боковом меню, чтобы увидеть аналитику.")
        if lottie_robot:
            st_lottie(lottie_robot, height=200)

# --- TAB 2: MASS CHECK ---
with tab_batch:
    target_df = None
    if uploaded_file:
        target_df = load_data(uploaded_file)
        
    if target_df is not None:
        st.write(f"Файл загружен. Готово к проверке {len(target_df)} транзакций.")
        
        # Проверяем режим работы
        has_class = 'Class' in target_df.columns
        if not has_class:
            st.info("📢 Режим: **Поиск неизвестных угроз** (Колонка 'Class' отсутствует, модель будет предсказывать мошенников)")
        else:
            st.success("📢 Режим: **Тестирование точности** (Найдена колонка 'Class', сравним предсказания с фактом)")

        if st.button("🚀 Проверить весь файл (Batch Processing)"):
            progress_bar = st.progress(0)
            try:
                # 1. Подготовка (Заполняем недостающие колонки нулями)
                check_df = target_df.copy()
                
                # Убираем Class, чтобы не мешал
                if 'Class' in check_df.columns:
                    check_df = check_df.drop('Class', axis=1)
                
                missing = set(model_columns) - set(check_df.columns)
                for c in missing: check_df[c] = 0
                check_df = check_df[model_columns] # Строгий порядок колонок
                
                # 2. Предсказание
                progress_bar.progress(30)
                probs = model.predict_proba(check_df)[:, 1]
                preds = (probs >= threshold).astype(int)
                progress_bar.progress(90)
                
                # 3. Запись результатов
                target_df['AI_Risk_Score'] = probs
                target_df['AI_Verdict'] = preds
                
                progress_bar.progress(100)
                st.balloons()
                
                fraud_found = target_df[target_df['AI_Verdict'] == 1]
                
                # Вывод результатов
                m1, m2 = st.columns(2)
                m1.metric("Найдено угроз", len(fraud_found), delta="ALERT" if len(fraud_found)>0 else "CLEAN", delta_color="inverse")
                m2.metric("Средний уровень риска", f"{probs.mean()*100:.2f}%")
                
                st.subheader("🚨 Найденные подозрительные транзакции")
                st.dataframe(fraud_found.sort_values('AI_Risk_Score', ascending=False).head(50).style.background_gradient(subset=['AI_Risk_Score'], cmap='Reds'))
                
                # ПОКАЗЫВАЕМ ТОЧНОСТЬ ТОЛЬКО ЕСЛИ БЫЛ CLASS
                if has_class:
                    st.markdown("---")
                    st.subheader("📉 Отчет о точности")
                    acc = accuracy_score(target_df['Class'], preds)
                    cm = confusion_matrix(target_df['Class'], preds)
                    st.metric("Accuracy", f"{acc:.2%}")
                    fig_cm = px.imshow(cm, text_auto=True, title="Матрица ошибок", x=['Pred Ok', 'Pred Fraud'], y=['True Ok', 'True Fraud'])
                    st.plotly_chart(fig_cm, use_container_width=True)
                
                # Скачивание
                csv = target_df.to_csv(index=False).encode('utf-8')
                st.download_button("💾 Скачать полный отчет (CSV)", csv, "fraud_report.csv", "text/csv")
                
            except Exception as e:
                st.error(f"Ошибка при обработке: {e}")
    else:
        st.info("Загрузите CSV файл для массовой проверки.")

# --- TAB 3: SHAP ---
with tab_ai:
    st.header("🧠 Интерпретация решений (SHAP)")
    st.info("Этот модуль показывает, какие именно признаки повлияли на решение AI.")
    
    if st.button("📊 Построить SHAP графики"):
        with st.spinner("Генерация отчета объяснимости..."):
            try:
                # 1. Поиск модели для объяснения
                estimator = None
                # Если это стекинг, ищем внутри него
                if hasattr(model, 'named_estimators_'):
                    # Приоритет: XGBoost -> CatBoost -> RF
                    for name in ['xgb', 'cat', 'lgbm', 'rf']:
                        if name in model.named_estimators_:
                            estimator = model.named_estimators_[name]
                            break
                
                # Если не нашли или модель простая - берем как есть
                if estimator is None:
                    if hasattr(model, 'estimators_'):
                        estimator = model.estimators_[0]
                    else:
                        estimator = model # Сама модель не стекинг

                # 2. Данные для фона
                if uploaded_file:
                    bg_df = load_data(uploaded_file)
                    
                    # Если есть Class, убираем его
                    if 'Class' in bg_df.columns:
                        bg_df = bg_df.drop('Class', axis=1)
                    
                    bg_df = bg_df.sample(min(50, len(bg_df))) # Берем сэмпл для скорости
                    
                    # Чистим данные под формат модели
                    bg_clean = pd.DataFrame(0, index=bg_df.index, columns=model_columns)
                    common = list(set(bg_df.columns) & set(model_columns))
                    bg_clean[common] = bg_df[common]
                else:
                    # Если файла нет, генерируем шум
                    bg_clean = pd.DataFrame(np.random.rand(50, len(model_columns)), columns=model_columns)

                # 3. Расчет SHAP values
                explainer = shap.TreeExplainer(estimator)
                shap_values = explainer.shap_values(bg_clean)
                
                # 4. Визуализация (Исправленная версия с plt.figure)
                st.markdown("### Визуализация важности признаков")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Вклад признаков (Summary Plot)**")
                    fig1 = plt.figure(figsize=(10, 6))
                    shap.summary_plot(shap_values, bg_clean, show=False)
                    st.pyplot(fig1)
                    plt.close(fig1) # Обязательно закрываем
                
                with col2:
                    st.markdown("**Топ важных признаков (Bar Chart)**")
                    fig2 = plt.figure(figsize=(10, 6))
                    shap.summary_plot(shap_values, bg_clean, plot_type="bar", show=False)
                    st.pyplot(fig2)
                    plt.close(fig2)
                    
            except Exception as e:
                st.error(f"Ошибка построения SHAP графиков: {e}")
                st.info("Подсказка: Убедитесь, что библиотеки версий XGBoost/CatBoost совпадают с теми, на которых обучалась модель.")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: grey;'>© 2026 AI Fraud Detection System | Powered by Stacking Ensemble Architecture</div>", unsafe_allow_html=True)