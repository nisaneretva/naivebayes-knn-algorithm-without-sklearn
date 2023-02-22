import pandas as pd
import itertools
from sklearn.preprocessing import MinMaxScaler # verilerin daha temiz olması ve daha iyi sonuc almak için normalizasyon işlemi için bu kütüphaneyi kullandım
scaler = MinMaxScaler()
dictionary={}
count=0
true_class = 0
false_class = 0

def class_hesapla(dictionary,test_class_name): #classların hesaplanması ve sınıf tahmini
    #print("class_hesapla")
    global true_class, false_class
    sorted_dic = {k: v for k, v in sorted(dictionary.items(), key = lambda v: v[1])} # en yakın 3 uzaklıgı secmek ıcın dict value degerine göre sıralanması
    #print("sorted dic: ",sorted_dic)
    sorted_three =dict(itertools.islice(sorted_dic.items(), 3)) # en kucuk 3 dict iteminin alınması
   
    # print(list(sorted_four.keys()))
    class_list=[]
    for i in list(sorted_three.keys()): # dict icindeki keyler indexleri tutmaktadır
        #print(df1[df1["Index"]==i]["Index"].values[0])
        #print(i)
        class_list.append(df1[df1["Index"]==i]["Class"].values[0]) # index degerine gore df1 deki feature gidilerek o verinin classı listeye atılmaktadır

    #print(class_list)
    cikan=max(class_list,key = class_list.count) #listede en çok tekrar eden degeri sınıfa atmaktadır
    #print("cikan: ", cikan)
    if cikan == test_class_name:
        true_class+=1 # her dogru deger icin +1
        #print("<<True>>  Cikan Deger:",cikan)
    else:
        false_class+=1 # her yanlıs deger icin skor tahminin de kullanılmak üzere +1 verilmektedir
        #print("<<False>>  Cikan Deger: "+cikan+" Gercek Deger: "+test_class_name)

def knn(df,exam,train_class,test_class):
    global dictionary
    global count
    global test_class_name
    #print("knn")
    for i in range(len(df)-2):  # test verisi tüm satırlarla karşılaştırmak icin dongu kurulması
        #k = input("Lutfen bir k int degeri giriniz.")
    
        distance = 0.0
        for j in range(len(test_class.columns)):  # test verisinin columnlarının train verileri ile karsılastırılması
        
            #print("train",train_class.iloc[i,j])
            #print("test",test_class.iloc[0,j])
            distance += (train_class.iloc[i,j]-test_class.iloc[0,j])**2 # oklid uzaklıgı için verilerin birbirinden çıkarılması ve karesinin alınması
            #print(distance)
        x =train_class.index.values[i] # çıkan train verilerinin index degerlerine göre dictionary'a atılması
        dictionary[x] = distance**0.5 # indekse oklid uzaklıgının atılması
        #print(dictionary)
       
        if(i==len(df)-3): 
            count+=1
           
            class_hesapla(dictionary, test_class_name) # tum train verileri ile test verisinin uzaklıkları hesaplandıktan sonra sınıf tahmini için burası çalışmaktadır
            exam = pd.DataFrame(df.iloc[count:(count+1),:]) # hesaplandıktan sonra kalan test verimiz traine katılarak 
            #print(exam)                                    # trainden başka bir test verisi gelmektedir ve bu tüm dataset test verisi oluncaya kadar tekrarlanmaktadır
            test_class_name = exam["Class"].values[0]
            # print(test_class_name)
            test_class=exam.drop("Class", axis=1)
            train_class = df.drop(test_class.index)
            #print(dictionary)
            dictionary={} 
            try:
                #print("return ettim")
                return knn(df,exam,train_class,test_class) #recursive olarak fonk kendini çağırmaktadır
            except:
                sonuc=(true_class/(true_class+false_class))*100 # hiçbir veri kalmadıgı takdirde skor hesaplaması yapılmıştır
                print("Skor Sonucu: ",sonuc)    

df = pd.read_excel(r"C:\Users\nisan\Downloads\Dry_Bean_Dataset.xlsx") #dosyanın okunması
df.iloc[:,0:-1] = scaler.fit_transform(df.iloc[:,0:-1].to_numpy()) #verilerin normalizasyonu
#df = df.iloc[0:10,:]
#df=df.sample(frac = 0.01, random_state = 42)
df1 = df.copy()  # burada indexler arasında gezinmek için index featureuolan bir df kopyası df1 olusturuyoruz
df1["Index"] = df.index
print(df)
print(df1)

# print(df.head())
# print(df.columns)
# print(df.describe())
# print(df.info())

# train = df.sample(frac = 0.7, random_state = 42) # %70 train
# test = df.drop(train.index) # %30 test
exam = pd.DataFrame(df.iloc[0:1,:]) # ilk verimizin alınması
test_class_name = exam["Class"].values[0] # ilk verimizin classının tutulması
# print(test_class_name)
test_class=exam.drop("Class", axis=1) # test verisinden class degerinin cıkarılması
train_class = df.drop(test_class.index) # train verisinden ilk verinin çıkarılması
# print(len(test_class.columns))
# print(train_class)
# print(test_class)
 
knn(df,exam,train_class,test_class) # knn hesaplama fonk cagirilmasi
