from django.db import models

class Request(models.Model):
    BOOL_CHOICES= [
        (1,'YES'),
        (0, 'NO')
    ]
    Applicant_Name = models.CharField(max_length=25, null=False)
    Gender = models.IntegerField(max_length=10, choices= [
        (0,'male'),
        (1,'female'),
    ])
    Married = models.IntegerField(max_length=3, choices=BOOL_CHOICES)
    Dependents = models.FloatField()
    Education = models.CharField(max_length=20, choices=[
        ('Graduate', 'Graduate'), 
        ('Not Graduate', 'Not Graduate')
        ])
    Self_Employed = models.IntegerField(max_length=3, choices=BOOL_CHOICES)
    ApplicantIncome = models.IntegerField(default=0)
    CoapplicantIncome = models.IntegerField(default=0)
    LoanAmount = models.IntegerField(default=0)
    Loan_Amount_Term = models.SmallIntegerField(default=0)
    Credit_History = models.IntegerField(max_length=10, choices=BOOL_CHOICES)
    Property_Area = models.CharField(max_length=20, choices=[
        ('Urban', 'Urban'),
        ('Rural', 'Rural'),
        ('Semiurban', 'Semiurban'),
    ])
    status = models.CharField(max_length=20, choices=[
        ('approved', 'Approved'), 
        ('rejected', 'Rejected')
        ])
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.applicant_name
