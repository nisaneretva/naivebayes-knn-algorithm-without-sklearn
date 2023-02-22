import numpy as np
import pandas as pd

def Accuracy(y, prediction): # score hesaplama
    prediction = list(prediction)
    score = 0
    
    for i, j in zip(list(y), prediction):
        if i == j:
            score += 1 #scorun başarı durumunda +1 deger alması ve bunun tum verilere bolunması
    return score / len(y)

df = pd.read_excel(r"C:\Users\nisan\Downloads\Dry_Bean_Dataset.xlsx") #dosyanın okunması
# print(df.head())
# print(df.columns)
# print(df.describe())
# print(df.info())
categorical = [var for var in df.columns if df[var].dtype=='O'] # kategorik verilerin bulunması
print('The categorical variables are :', categorical)
numerical = [var for var in df.columns if df[var].dtype!='O'] # numerik verilerin bulunması
print('The numerical variables are:', numerical)

# TRAIN TEST SPLIT
train = df.sample(frac = 0.7, random_state = 42) # %70 train
test = df.drop(train.index) # %30 test
y_train = train["Class"]  # tahmin edilcek train classlarının tutulması
x_train = train.drop("Class", axis = 1) # tahmin edilecek train classlarının drop edilmesi
# print(x_train.index)
y_test = test["Class"] # tahmin edilcek test classlarının tutulması
x_test = test.drop("Class", axis = 1) # tahmin edilecek test classlarının drop edilmesi
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

means = train.groupby(["Class"]).mean() # ortalama hesabı
var = train.groupby(["Class"]).var() # varyans hesabı
prior = (train.groupby("Class").count() / len(train)).iloc[:,1] # h ın ilk olasılığı
classes = train['Class'].unique().tolist() # class isimleri
# print(means)
# print(var)
# print(prior)

def predict(train_x):
    global count
    predics = []
    for i in train_x.index: # Verilerin indexini tutar
        #print(train_x.index)
        sinif_olasilik = [] #sınıflar arası olasılıkların tutulması icin
        instance = train_x.loc[i]
        #print("instance",instance)
        for cls in classes: # classlar arasında dolasma
            ozel_olasilik = [] #ozellikler arası olasilikların tutualması icin
            ozel_olasilik.append((prior[cls])) # class sayısının toplam veriye bölümü (prior) eklenmesi
            for i in train_x.columns: # columnlar arasında dolasma
                #print("col",col)
                x = instance[i]
                mean = means[i].loc[cls] # o sınıfa ait columnların ortalama hesabı 
                #print("mean",mean)
                sta_sap = var[i].loc[cls]**0.5 #standart sapma
                #print("cls",cls)
                #(1/(sta.sap * 2pi**0.5)) * (e**(-0.5 (((x-ort)/(sta.sap))**2))
                olasilik = (np.e ** (-0.5 * ((x - mean)/sta_sap) ** 2)) / (sta_sap * np.sqrt(2 * np.pi)) #sayısal verilerde naive bayes hesabı
                if olasilik == 0:
                    olasilik = 1/len(train) 
                ozel_olasilik.append(olasilik)
                #print("ozellik olasiligi",ozel_olasilik)
            top_olasilik = np.prod(ozel_olasilik) # posterior
            sinif_olasilik.append(top_olasilik)
        # En buyuk posterior belirlenmesi ve predictiona eklenmesi    
        maxi = sinif_olasilik.index(max(sinif_olasilik))  # olasiligi max olan sınıfın indexinin tutulması
        prediction = classes[maxi] 
        predics.append(prediction)
    return predics

predict_train = predict(x_train)
predict_test = predict(x_test)
print("\n")
print("Skor Sonucu: ",Accuracy(y_train, predict_train))
print("Skor Sonucu: ",Accuracy(y_test, predict_test))

