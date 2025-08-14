import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

class Modelo:
    def __init__(self):
        self.df = None
        self.models = {}

    def CarregarDataset(self, path):
        column_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
        self.df = pd.read_csv(path, names=column_names)

    def TratamentoDeDados(self):
        """
        Realiza o pré-processamento dos dados carregados.
        """
        # Verificar dados ausentes
        print("Dados ausentes por coluna:")
        print(self.df.isnull().sum())
        
        # Remover qualquer linha com dados ausentes (caso exista)
        self.df.dropna(inplace=True)

        # Codificar a coluna Species (de string para valores numéricos)
        label_encoder = LabelEncoder()
        self.df['Species'] = label_encoder.fit_transform(self.df['Species'])
        self.label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

        # Escalar os dados para melhorar o desempenho dos modelos
        scaler = StandardScaler()
        self.df.iloc[:, :-1] = scaler.fit_transform(self.df.iloc[:, :-1])

    def Treinamento(self):
        """
        Treina diferentes modelos de machine learning e compara seus desempenhos.
        """
        # Separar características (X) e alvo (y)
        X = self.df.iloc[:, :-1]
        y = self.df['Species']

        # Dividir dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Modelos a serem treinados
        models = {
            'SVM': SVC(kernel='linear', random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42),
        }

        # Treinar cada modelo e avaliar desempenho
        for name, model in models.items():
            # Validação cruzada durante o treinamento
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            print(f"{name} - Validação Cruzada (média): {scores.mean():.2f}")
            
            # Treinamento no conjunto de treino
            model.fit(X_train, y_train)

            # Avaliar no conjunto de teste
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"{name} - Acurácia no conjunto de teste: {accuracy:.2f}")
            print(classification_report(y_test, y_pred, target_names=self.label_mapping.keys()))
            
            # Salvar modelo treinado para uso futuro
            self.models[name] = model

    def Teste(self):
        """
        Testa o desempenho dos modelos treinados em novos dados (se necessário).
        """
        # Testar um modelo específico (por exemplo, SVM)
        if 'SVM' in self.models:
            model = self.models['SVM']
            # Simular testes com novos dados (opcional, criar dados ou carregar outro dataset)
            print("Modelo SVM está treinado e pronto para teste com novos dados!")

    def Train(self):
        """
        Fluxo completo de execução: Carregamento, tratamento, treinamento e teste.
        """
        self.CarregarDataset("iris.data")
        self.TratamentoDeDados()
        self.Treinamento()
        self.Teste()

# Execução do programa
if __name__ == "__main__":
    modelo = Modelo()
    modelo.Train()
