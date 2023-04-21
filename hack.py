import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import random
import mysql.connector as my
con = my.connect(user = 'root',password = '1234', host = 'localhost', database = 'security' )
cursor = con.cursor()


print("\n-----Welcome to Health Vault-----")
print('\n')

a = input('Are you a new user?(Y/N) : ')
if a.lower() == 'y':
    d = input('Enter Patient ID: ')
    b = input('Enter your password: ') 
    c = input('Confirm your password: ')
    
    if b == c :
        print('Password Confirmed!')
        ab = 'insert into passwords (p_id,password) values(%s,%s)'
        ac = (d,b,)
        cursor.execute(ab,ac)
        con.commit()
    
    else:
        print('Password does not match...')
        tr = input("Would you like to try again?(Y/N) : ")
        count = 1
       
        if tr.lower() == 'y':
            2
            while count < 2:
                d = input('Enter Patient ID: ')
                b = input('Enter your password: ')
                c = input('Confirm your password: ')
                
                if b == c :
                    print('Password Confirmed!')
                    ab = 'insert into passwords (platform,password) values(%s,%s)'
                    ac = (d,b,)
                    cursor.execute(ab,ac)
                    con.commit()
                    break
               
                elif b!=c:
                    print("Password does not match, Exceeded number of chances!!")
                    count +=1

       


    #Inserting patient data 
    print('\n')
    de = input("Renter patient ID: ")
    be = input("Enter patient name: ")
    ce = int(input("Enter the age: "))
    ge = input("Enter the gender: ")
    fe = int(input("Enter the phone: "))

    ab = 'insert into patients (p_id,patient_name,age,Gender,phone_no) values(%s,%s,%s,%s,%s)'
    ac = (de,be,ce,ge,fe,)
    cursor.execute(ab,ac)
    con.commit() 
    print("Account succesfully created!\nPlease refresh the page and login again to get further access\n")    





    
elif a.lower() == 'n':
    usern = input('Enter patient ID: ')
    pwdu = input('Enter patients password: ')
    cursor.execute('select password from passwords where p_id like (%s)',(usern,))
    result = cursor.fetchall()
    for i in result:
        for j in i:
            if j == pwdu:
                print('Login successfully completed')
            else:
                print('Incorrect password! \nCheck your password and username ')
                break
            break
        break


    print("\n\n--/Control panel\-- \n\n1.Display patient details \n2.Book an appointment \n3.AI consultation")
    pri = int(input())


    
    if pri == 1:
        print("\n")
        cursor.execute('select * from patients where p_id = (%s)',(usern,))
        result = cursor.fetchall()
        print("--------------------")
        for i in result:
            for j in i:
                print(j)
        print("--------------------")



    if pri == 2:
        print('\n')
        cursor.execute('select * from doctor')
        result = cursor.fetchall()
        print("--------------------")

        count = 0
        
        for i in result:
            for j in i:
                print(j," ", end = "")
                count +=1
                if count <4:
                    continue
                else:
                    print("\n")
                    count = 0
                    continue
        print("--------------------\n")

        

        specialist = input("Enter the specialist that you want to consult: ")
        dept = input("Enter the deparment: ")
        cursor.execute("Insert into booking (p_id, doctor, department) values(%s,%s,%s)",(usern,specialist,dept))
        con.commit()    
        print("Appointment successfully booked.")
        random_no = random.randint(1,100)
        print("Your Token number is : ", random_no)
        print("\nThank you for using HealthVault!\n")

    if pri == 3:
        DATA_PATH = "testing.csv"
        data = pd.read_csv(DATA_PATH)
 
        # Checking whether the dataset is balanced or not
        disease_counts = data["prognosis"].value_counts()
        temp_df = pd.DataFrame({
            "Disease": disease_counts.index,
            "Counts": disease_counts.values
        })
        encoder = LabelEncoder()
        data["prognosis"] = encoder.fit_transform(data["prognosis"])
        X = data.iloc[:,:-1]
        y = data.iloc[:, -1]
        X_train, X_test, y_train, y_test =train_test_split(X, y, test_size = 0.2, random_state = 24)
        def cv_scoring(estimator, X, y):
            return accuracy_score(y, estimator.predict(X))
 
        # Initializing Models
        models = {"SVC":SVC(), "Gaussian NB":GaussianNB(), "Random Forest":RandomForestClassifier(random_state=18)}
        svm_model = SVC()
        svm_model.fit(X_train, y_train)
        preds = svm_model.predict(X_test)

        # Training and testing Naive Bayes Classifier
        nb_model = GaussianNB()
        nb_model.fit(X_train, y_train)
        preds = nb_model.predict(X_test)
 
        # Training and testing Random Forest Classifier
        rf_model = RandomForestClassifier(random_state=18)
        rf_model.fit(X_train, y_train)
        preds = rf_model.predict(X_test)

        # Training and testing SVM Classifier
        svm_model = SVC()
        svm_model.fit(X_train, y_train)
        preds = svm_model.predict(X_test)
 
        # Training and testing Naive Bayes Classifier
        nb_model = GaussianNB()
        nb_model.fit(X_train, y_train)
        preds = nb_model.predict(X_test)
        final_svm_model = SVC()
        final_nb_model = GaussianNB()
        final_rf_model = RandomForestClassifier(random_state=18)
        final_svm_model.fit(X, y)
        final_nb_model.fit(X, y)
        final_rf_model.fit(X, y)

        test_data = pd.read_csv("testing.csv").dropna(axis=1)
 
        test_X = test_data.iloc[:, :-1]
        test_Y = encoder.transform(test_data.iloc[:, -1])
 
        # Making prediction by take mode of predictions
        # made by all the classifiers
        svm_preds = final_svm_model.predict(test_X)
        nb_preds = final_nb_model.predict(test_X)
        rf_preds = final_rf_model.predict(test_X)
        final_preds = [mode([i,j,k])[0][0] for i,j,k in zip(svm_preds, nb_preds, rf_preds)]
        symptoms = X.columns.values
 
        # Creating a symptom index dictionary to encode the
        # input symptoms into numerical form
        symptom_index = {}
        for index, value in enumerate(symptoms):
            symptom = " ".join([i.capitalize() for i in value.split("_")])
            symptom_index[symptom] = index
 
        data_dict = {
            "symptom_index":symptom_index,
            "predictions_classes":encoder.classes_
        }
 
        # Defining the Function
        # Input: string containing symptoms separated by commas
        # Output: Generated predictions by models
        def predictDisease():
            print("\n\n\n")
            symptoms = input("State your symptoms : ")
            symptoms = symptoms.split(",")
     
            # creating input data for the models
            input_data = [0] * len(data_dict["symptom_index"])
            for symptom in symptoms:
                index = data_dict["symptom_index"][symptom]
                input_data[index] = 1
         
            # reshaping the input data and converting it
            # into suitable format for model predictions
            input_data = np.array(input_data).reshape(1,-1)
     
            # generating individual outputs
            rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
            nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
            svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]
     
            # making final prediction by taking mode of all predictions
            final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])[0][0]
            predictions = {
                "rf_model_prediction": rf_prediction,
                "naive_bayes_prediction": nb_prediction,
                "svm_model_prediction": svm_prediction,
                "final_prediction":final_prediction
            }
            return predictions
 
        # Testing the function
        cut = predictDisease()
        mylst = list(cut.values())
        wordfreq = {}

        for word in mylst:
            if word in wordfreq:
                wordfreq[word] += 1
            else:
             wordfreq[word] = 1

        mostcmnword = max(wordfreq, key=wordfreq.get)
        print("\n\n\n")
        print("The most probable disease for the mentioned syptoms is: ",mostcmnword)
        print("\n\n\n")