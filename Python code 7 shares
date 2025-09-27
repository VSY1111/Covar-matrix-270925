import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Скачиваем данные по 7 популярным акциям за последние 60 дней
tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NFLX']

# Получаем данные
end_date = datetime.now()
start_date = end_date - timedelta(days=60)

data = {}
for ticker in tickers:
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        data[ticker] = stock_data['Close']
        print(f"Данные для {ticker} загружены")
    except:
        print(f"Ошибка загрузки данных для {ticker}")

# Создаем DataFrame
df = pd.DataFrame(data)

# Удаляем строки с пропущенными значениями
df_clean = df.dropna()

print(f"\nДанные по ценам закрытия:")
print(df_clean.head())
print(f"\nРазмер данных: {df_clean.shape}")

# Рассчитываем дневную доходность (процентное изменение)
returns = df_clean.pct_change().dropna()

print(f"\nДоходности:")
print(returns.head())

# Рассчитываем ковариационную матрицу
cov_matrix = returns.cov()

print(f"\nКовариационная матрица:")
print(cov_matrix)

# Визуализируем ковариационную матрицу
import matplotlib.pyplot as plt
import seaborn as sns 

plt.figure(figsize=(10, 8))
sns.heatmap(cov_matrix, annot=True, fmt='.6f', cmap='coolwarm', 
            square=True, cbar_kws={"shrink": .8})
plt.title('Ковариационная матрица доходностей акций')
plt.tight_layout()
plt.show()

# Альтернативный способ: расчет вручную с помощью numpy
print("\n" + "="*50)
print("Ручной расчет ковариационной матрицы:")

# Центрируем данные (вычитаем среднее)
returns_centered = returns - returns.mean()

# Ручной расчет ковариационной матрицы
n = len(returns_centered)
manual_cov_matrix = returns_centered.T @ returns_centered / (n - 1)

print("Результат ручного расчета:")
print(manual_cov_matrix)

# Сравнение результатов
print(f"\nМаксимальная разница между методами: {np.max(np.abs(cov_matrix - manual_cov_matrix))}")
# добавил строчку 27092025 
