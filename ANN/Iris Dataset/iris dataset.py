import pandas as pd

class Tester:
    def loadData(self):
        self.datas = pd.read_excel('Iris.xls')
        self.x = self.datas.iloc[:,1:4]
        self.y = self.datas.iloc[:,4:]

    def trainAndTest(self):
        from sklearn.cross_validation import train_test_split
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.33, random_state=0)
    
    def scale(self):
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.x_train)
        self.X_test = sc.transform(self.x_test)
    
    def lgr(self):
        print("\nLogistic Regression Results: ")
        from sklearn.linear_model import LogisticRegression
        lgr = LogisticRegression(random_state = 0)
        lgr.fit(self.X_train, self.y_train)
        self.y_pred = lgr.predict(self.X_test)
        self.analyze()
    
    def knn(self):
        print("\nKNN Results: ")
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
        knn.fit(self.X_train, self.y_train)
        self.y_pred = knn.predict(self.X_test)
        self.analyze()
    
    def svc(self):
        print("\nSVC Results: ")
        from sklearn.svm import SVC
        svc = SVC(kernel='linear') # linear, sigmoid, poly (1)
        svc.fit(self.X_train, self.y_train)
        self.y_pred = svc.predict(self.X_test)
        self.analyze()
        
    def nb(self):
        print("\nNaive Bayes Results: ")
        from sklearn.naive_bayes import GaussianNB
        gnb = GaussianNB()
        gnb.fit(self.X_train, self.y_train)
        self.y_pred = gnb.predict(self.X_test)
        self.analyze()
        
    def dtc(self):
        print("\nDecision Tree Classification Results: ")
        from sklearn.tree import DecisionTreeClassifier
        dtc = DecisionTreeClassifier(criterion="gini")
        dtc.fit(self.X_train, self.y_train)
        self.y_pred = dtc.predict(self.X_test)
        self.analyze()
        
    def rfc(self):
        print("\nRandom Forest Classification Results: ")
        from sklearn.ensemble import RandomForestClassifier
        rfc = RandomForestClassifier(n_estimators=5, criterion="entropy")
        rfc.fit(self.X_train, self.y_train)
        self.y_pred = rfc.predict(self.X_test)
        self.analyze()
    
    def analyze(self):
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        print(cm)
        
tester = Tester()
tester.loadData()
tester.trainAndTest()
tester.scale()
tester.lgr()
tester.knn()
tester.svc()
tester.nb()
tester.dtc()
tester.rfc()