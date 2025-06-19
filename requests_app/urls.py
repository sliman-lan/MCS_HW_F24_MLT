from django.urls import path
from .views import request_list, request_create, request_update, request_delete, home, report

urlpatterns = [
    path('', home, name='home'),
    path('report/', report, name='EDA_REPORT'),
    path('requests/', request_list, name='request_list'),
    path('new/', request_create, name='request_create'),
    path('edit/<int:pk>/', request_update, name='request_update'),
    path('delete/<int:pk>/', request_delete, name='request_delete'),
]
