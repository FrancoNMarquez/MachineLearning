import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# --- 1. Simulación de Datos  ---
np.random.seed(42) # para reproducibilidad

# 1. Entidad VEHICULO
n_vehiculos = 5000
marcas = ['Toyota', 'Ford', 'Chevrolet', 'Volkswagen', 'Honda', 'Nissan', 'Hyundai', 'Fiat', 'Peugeot', 'Renault', 'Mercedes-Benz', 'Audi']
colores = ['Blanco', 'Negro', 'Gris', 'Rojo', 'Azul', 'Plata', 'Verde']

df_vehiculo = pd.DataFrame({
    'nro_chasis': [f'CHASIS_{i:04d}' for i in range(1, n_vehiculos + 1)],
    'anio_fabricacion': np.random.randint(2010, 2025, n_vehiculos),
    'marca': np.random.choice(marcas, n_vehiculos),
    'color': np.random.choice(colores, n_vehiculos),
    'precio': np.random.randint(18000, 65000, n_vehiculos), 
})

# 2. Entidad CLIENTE
n_clientes = 2000
localidades = [f'Loc_{i}' for i in range(1, 25)] 
sexos = ['M', 'F', 'X']

df_cliente = pd.DataFrame({
    'cuit': [f'{np.random.randint(20, 35)}-{np.random.randint(10000000, 99999999)}-{np.random.randint(0, 10)}' for _ in range(n_clientes)],
    'direccion_calle': [f'Calle {i}' for i in range(1, n_clientes + 1)],
    'direccion_numero': np.random.randint(1, 2000, n_clientes),
    'localidad': np.random.choice(localidades, n_clientes),
    'telefono': [f'11{np.random.randint(10000000, 99999999)}' for _ in range(n_clientes)],
    'email': [f'cliente{i}@example.com' for i in range(1, n_clientes + 1)],
    'nombre': [f'Nombre_{i}' for i in range(1, n_clientes + 1)],
    'apellido': [f'Apellido_{i}' for i in range(1, n_clientes + 1)],
    'edad': np.random.randint(18, 70, n_clientes),
    'ingresos': np.random.randint(30000, 150000, n_clientes)
})
df_cliente = df_cliente.drop_duplicates(subset=['cuit'])

# 3. Entidad FACTURA
n_facturas = 7000
df_factura = pd.DataFrame({
    'nro_factura': [f'FAC_{i:05d}' for i in range(1, n_facturas + 1)],
    'fecha': pd.to_datetime('2020-01-01') + pd.to_timedelta(np.random.randint(0, 1825, n_facturas), unit='D'), # 5 años de data
    'comision': np.round(np.random.uniform(750, 3000, n_facturas), 2),
    'patente': [f'AA{np.random.randint(100, 999)}BB' for _ in range(n_facturas)],
    'cuit_cliente': np.random.choice(df_cliente['cuit'], n_facturas),
})
df_factura = df_factura.drop_duplicates(subset=['nro_factura'])


# 4. Entidad VENTA
df_venta = pd.DataFrame({
    'id_venta': range(1, n_facturas + 1),
    'pago': (pd.to_datetime('2020-01-01') + pd.to_timedelta(np.random.randint(0, 1825, n_facturas), unit='D')).astype(str) + ',' + np.random.randint(1000, 9999, n_facturas).astype(str),
    'cuit_cliente': np.random.choice(df_cliente['cuit'], n_facturas),
    'nro_chasis': np.random.choice(df_vehiculo['nro_chasis'], n_facturas),
    'nro_factura': np.random.choice(df_factura['nro_factura'], n_facturas),
})
df_venta = df_venta.drop_duplicates(subset=['id_venta'])


# --- 2. Integración de Datos ---
df_merged = df_venta.merge(df_vehiculo, on='nro_chasis', how='left')
df_merged = df_merged.merge(df_cliente, left_on='cuit_cliente', right_on='cuit', how='left')
df_merged = df_merged.merge(df_factura, on='nro_factura', how='left', suffixes=('_venta', '_factura'))

df_merged['Fecha_Venta'] = df_merged['pago'].apply(lambda x: x.split(',')[0] if pd.notnull(x) else None)
df_merged['Fecha_Venta'] = pd.to_datetime(df_merged['Fecha_Venta'])

# --- 3. Ingeniería de Características ---
df_merged['Anio_Venta'] = df_merged['Fecha_Venta'].dt.year
df_merged['Mes_Venta'] = df_merged['Fecha_Venta'].dt.month
df_merged['Trimestre_Venta'] = df_merged['Fecha_Venta'].dt.quarter
df_merged['Dia_Semana_Venta'] = df_merged['Fecha_Venta'].dt.dayofweek

df_merged['Antiguedad_Vehiculo_Al_Vender'] = df_merged['Anio_Venta'] - df_merged['anio_fabricacion']
df_merged = df_merged[df_merged['Antiguedad_Vehiculo_Al_Vender'] >= 0].copy()

# --- Agregación para obtener la variable objetivo: Ventas por Marca por Año ---
df_ventas_anuales_marca = df_merged.groupby(['Anio_Venta', 'marca']).size().reset_index(name='Cantidad_Ventas')

df_precio_marca_anual = df_merged.groupby(['Anio_Venta', 'marca'])['precio'].mean().reset_index(name='Precio_Promedio_Marca_Anual')
df_comision_marca_anual = df_merged.groupby(['Anio_Venta', 'marca'])['comision'].mean().reset_index(name='Comision_Promedio_Marca_Anual')
df_edad_cliente_marca_anual = df_merged.groupby(['Anio_Venta', 'marca'])['edad'].mean().reset_index(name='Edad_Promedio_Cliente_Marca_Anual')


# Crear la tabla de características para el entrenamiento
features_df = df_ventas_anuales_marca.copy()
features_df['Anio_Venta_Anterior'] = features_df['Anio_Venta'] - 1
features_df = features_df.rename(columns={'Cantidad_Ventas': 'Cantidad_Ventas_Anio_Actual'})

features_df = features_df.merge(
    df_precio_marca_anual.rename(columns={'Anio_Venta': 'Anio_Venta_Anterior', 'Precio_Promedio_Marca_Anual': 'Precio_Promedio_Marca_Anio_Anterior'}),
    on=['Anio_Venta_Anterior', 'marca'],
    how='left'
)
features_df = features_df.merge(
    df_comision_marca_anual.rename(columns={'Anio_Venta': 'Anio_Venta_Anterior', 'Comision_Promedio_Marca_Anual': 'Comision_Promedio_Marca_Anio_Anterior'}),
    on=['Anio_Venta_Anterior', 'marca'],
    how='left'
)
features_df = features_df.merge(
    df_edad_cliente_marca_anual.rename(columns={'Anio_Venta': 'Anio_Venta_Anterior', 'Edad_Promedio_Cliente_Marca_Anual': 'Edad_Promedio_Cliente_Marca_Anio_Anterior'}),
    on=['Anio_Venta_Anterior', 'marca'],
    how='left'
)
features_df['Cantidad_Ventas_Anio_Anterior'] = features_df.groupby('marca')['Cantidad_Ventas_Anio_Actual'].shift(1)

df_model_ready = features_df.dropna(subset=[
    'Cantidad_Ventas_Anio_Anterior',
    'Precio_Promedio_Marca_Anio_Anterior',
    'Comision_Promedio_Marca_Anio_Anterior',
    'Edad_Promedio_Cliente_Marca_Anio_Anterior'
]).copy()

X = df_model_ready[[
    'marca',
    'Anio_Venta_Anterior',
    'Cantidad_Ventas_Anio_Anterior',
    'Precio_Promedio_Marca_Anio_Anterior',
    'Comision_Promedio_Marca_Anio_Anterior',
    'Edad_Promedio_Cliente_Marca_Anio_Anterior'
]]
y = df_model_ready['Cantidad_Ventas_Anio_Actual']

# --- 4. Preparación Final para el Modelo ---

y_scaler = StandardScaler() #Escalo los datos

numerical_cols = [
    'Anio_Venta_Anterior',
    'Cantidad_Ventas_Anio_Anterior',
    'Precio_Promedio_Marca_Anio_Anterior',
    'Comision_Promedio_Marca_Anio_Anterior',
    'Edad_Promedio_Cliente_Marca_Anio_Anterior'
]
categorical_cols = ['marca']

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='passthrough'
)

train_data_end_year = df_model_ready['Anio_Venta'].max() - 1
X_train = X[df_model_ready['Anio_Venta'] <= train_data_end_year]
y_train = y[df_model_ready['Anio_Venta'] <= train_data_end_year]

X_test = X[df_model_ready['Anio_Venta'] == df_model_ready['Anio_Venta'].max()]
y_test = y[df_model_ready['Anio_Venta'] == df_model_ready['Anio_Venta'].max()]

print(f"Dimensiones de X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Dimensiones de X_test: {X_test.shape}, y_test: {y_test.shape}")

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))

# --- 5. Construcción y Entrenamiento del Modelo ---
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_processed.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1) # Capa de salida para regresión lineal
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

history = model.fit(
    X_train_processed, y_train_scaled, 
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

print("\nModelo entrenado con éxito.")

# --- 6. Evaluación del Modelo ---
y_pred_test_scaled = model.predict(X_test_processed).flatten()

y_pred_test = y_scaler.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).flatten()

rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test = r2_score(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)

print(f"\nEvaluación en el conjunto de PRUEBA:")
print(f"RMSE (Root Mean Squared Error): {rmse_test:.2f}")
print(f"R^2 (Coeficiente de Determinación): {r2_test:.2f}")
print(f"MAE (Mean Absolute Error): {mae_test:.2f}")

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Loss de Entrenamiento')
plt.plot(history.history['val_loss'], label='Loss de Validación')
plt.title('Pérdida durante el entrenamiento de la RNA')
plt.xlabel('Época')
plt.ylabel('Pérdida (MSE)')
plt.legend()
plt.grid(True)
plt.show()

# --- 7. Predicción para el Próximo Año (2025 en este caso de simulación) ---
last_historical_year = df_ventas_anuales_marca['Anio_Venta'].max()

data_for_prediction = df_ventas_anuales_marca[df_ventas_anuales_marca['Anio_Venta'] == last_historical_year].copy()

data_for_prediction = data_for_prediction.rename(columns={
    'Anio_Venta': 'Anio_Venta_Anterior',
    'Cantidad_Ventas': 'Cantidad_Ventas_Anio_Anterior'
})

data_for_prediction = data_for_prediction.merge(
    df_precio_marca_anual.rename(columns={'Anio_Venta': 'Anio_Venta_Anterior_Merge', 'Precio_Promedio_Marca_Anual': 'Precio_Promedio_Marca_Anio_Anterior'}),
    left_on=['Anio_Venta_Anterior', 'marca'],
    right_on=['Anio_Venta_Anterior_Merge', 'marca'],
    how='left'
).drop(columns='Anio_Venta_Anterior_Merge')

data_for_prediction = data_for_prediction.merge(
    df_comision_marca_anual.rename(columns={'Anio_Venta': 'Anio_Venta_Anterior_Merge', 'Comision_Promedio_Marca_Anual': 'Comision_Promedio_Marca_Anio_Anterior'}),
    left_on=['Anio_Venta_Anterior', 'marca'],
    right_on=['Anio_Venta_Anterior_Merge', 'marca'],
    how='left'
).drop(columns='Anio_Venta_Anterior_Merge')

data_for_prediction = data_for_prediction.merge(
    df_edad_cliente_marca_anual.rename(columns={'Anio_Venta': 'Anio_Venta_Anterior_Merge', 'Edad_Promedio_Cliente_Marca_Anual': 'Edad_Promedio_Cliente_Marca_Anio_Anterior'}),
    left_on=['Anio_Venta_Anterior', 'marca'],
    right_on=['Anio_Venta_Anterior_Merge', 'marca'],
    how='left'
).drop(columns='Anio_Venta_Anterior_Merge')

prediction_year = last_historical_year + 1
data_for_prediction['Anio_Prediccion'] = prediction_year

X_predict = data_for_prediction[[
    'marca',
    'Anio_Venta_Anterior',
    'Cantidad_Ventas_Anio_Anterior',
    'Precio_Promedio_Marca_Anio_Anterior',
    'Comision_Promedio_Marca_Anio_Anterior',
    'Edad_Promedio_Cliente_Marca_Anio_Anterior'
]]

X_predict_processed = preprocessor.transform(X_predict)

predicted_sales_raw_scaled = model.predict(X_predict_processed)

predicted_sales = y_scaler.inverse_transform(predicted_sales_raw_scaled.reshape(-1, 1)).flatten()
predicted_sales = np.maximum(0, predicted_sales) # Asegurar que no haya ventas negativas después de desescalar

prediction_results = pd.DataFrame({
    'Marca': X_predict['marca'],
    'Anio_Prediccion': prediction_year,
    'Ventas_Predichas': predicted_sales
})

prediction_results = prediction_results.sort_values(by='Ventas_Predichas', ascending=False)

print(f"\n--- Predicción de Ventas por Marca para el año {prediction_year} ---")
print(prediction_results)

marca_mas_vendida = prediction_results.iloc[0]['Marca']
ventas_marca_mas_vendida = prediction_results.iloc[0]['Ventas_Predichas']

print(f"\nLa marca de autos que más se va a vender el próximo año ({prediction_year}) es: **{marca_mas_vendida}** con {ventas_marca_mas_vendida:.2f} unidades predichas.")

plt.figure(figsize=(12, 7))
sns.lineplot(data=df_ventas_anuales_marca, x='Anio_Venta', y='Cantidad_Ventas', hue='marca', marker='o')
sns.scatterplot(data=prediction_results, x='Anio_Prediccion', y='Ventas_Predichas', hue='Marca', s=200, marker='X', legend=False, zorder=5)

for index, row in prediction_results.iterrows():
    plt.text(row['Anio_Prediccion'], row['Ventas_Predichas'] + 0.5, f"{row['Marca']}: {row['Ventas_Predichas']:.0f}",
             horizontalalignment='center', fontsize=9, color='red')

plt.title('Ventas Históricas y Predicción de Ventas por Marca')
plt.xlabel('Año')
plt.ylabel('Cantidad de Ventas')
plt.xticks(np.arange(df_ventas_anuales_marca['Anio_Venta'].min(), prediction_year + 1, 1))
plt.grid(True)
plt.legend(title='Marca', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()