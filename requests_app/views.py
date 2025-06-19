import joblib
import numpy as np
from django.shortcuts import render, redirect
from .models import Request
from .forms import RequestForm
from .preprocessing import preprocess_data  # Import the preprocessing function

# Load the trained model
model = joblib.load('ml_model/trained_randomforest_model.pkl')  # Update the path to your model file

def home(request):
    return render(request, 'home.html')

def report(request):
    return render(request, 'EDA_Report.html')

def request_list(request):
    requests = Request.objects.all()
    return render(request, 'request_list.html', {'requests': requests})

def request_create(request):
    if request.method == 'POST':
        form = RequestForm(request.POST)
        if form.is_valid():
            # Save the request first
            new_request = form.save()

            # Preprocess the input data
            processed_data = preprocess_data(new_request.Applicant_Name, new_request.Gender,new_request.Married,new_request.Dependents, new_request.Education, new_request.Self_Employed, new_request.ApplicantIncome, new_request.CoapplicantIncome,new_request.LoanAmount,new_request.Loan_Amount_Term, new_request.Credit_History, new_request.Property_Area)

            # Make a prediction
            prediction = model.predict(processed_data)

            # Update the request with the prediction result
            new_request.status = 'approved' if prediction[0] == 1 else 'rejected'  # Adjust based on your model's output
            new_request.save()

            return redirect('request_list')
    else:
        form = RequestForm()
    return render(request, 'request_form.html', {'form': form})

def request_update(request, pk):
    request_instance = Request.objects.get(pk=pk)
    if request.method == 'POST':
        form = RequestForm(request.POST, instance=request_instance)
        if form.is_valid():
            form.save()
            return redirect('request_list')
    else:
        form = RequestForm(instance=request_instance)
    return render(request, 'request_form.html', {'form': form})

def request_delete(request, pk):
    request_instance = Request.objects.get(pk=pk)
    request_instance.delete()
    return redirect('request_list')
