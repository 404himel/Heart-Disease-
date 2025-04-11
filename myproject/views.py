from django.shortcuts import render
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
def home(request):
    return render(request,'home.html')
def result(request):
    model_path = r'C:\Users\Hp\OneDrive\Desktop\heart_failure\myproject\rf.pkl'

    with open(model_path,'rb') as file:
        model = joblib.load(file)

    if request.method=='GET':
        try:
            mylist = [request.GET[f'input{i}'] for i in range(1, 12)]
            print("Form data:", mylist)
        except Exception as e:
            print("Error getting input:", e)
            mylist = []

        if mylist:
            mylist = np.array(mylist).reshape(1, -1)
            print("data as std:", mylist)
            ans = model.predict(mylist)

            if ans==0:
                ans="You are not in risk"
            else:
                ans="You are on risk"
        
            return render(request, 'result.html', {'ans': ans})



    return render(request,'home.html')
