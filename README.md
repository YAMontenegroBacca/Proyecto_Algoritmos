Estructura de la carpeta

Nuevo_algoritmo/
│
├── env/                   # Entorno virtual
├── data/                  # Carpeta para tus archivos .mtx, .tsv, etc.
├── src/                   # Carpeta para el código fuente
│   ├── __init__.py        # Permite que Python reconozca esta carpeta como un paquete
│   └── main.py            # Script principal
├── requirements.txt       # Dependencias del proyecto
├── .gitignore             # Archivo para ignorar carpetas/archivos innecesarios
└── README.md              # Descripción del proyecto


Para instalar las dependencias en un entorno como un cluster ejecuta el siguiente comando:
pip install -r requirements.txt

Para ejecutar el algoritmo con los datos de prueba reducidos
python src/main.py --matrix_file datos/matrix_reduced.mtx --barcodes_file datos/barcodes_reduced.tsv --genes_file datos/genes.tsv