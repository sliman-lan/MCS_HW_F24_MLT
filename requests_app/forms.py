from django import forms
from .models import Request

class RequestForm(forms.ModelForm):
    class Meta:
        model = Request
        fields = ['Applicant_Name','Gender', 'Married', 'Dependents', 'Education',
                  'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome',
                  'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
                  'Property_Area' 
                  ]
