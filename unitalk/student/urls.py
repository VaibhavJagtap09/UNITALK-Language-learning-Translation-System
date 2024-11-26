from django.urls import path, include 
from .import views 
from django.contrib.auth import views as auth_views

urlpatterns = [

    path('', views.register ),
    path('home/', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('register/', views.register, name='register'),
    path('add_student/', views.add_student, name='add_student'),
    path('login/', views.login, name='login'),
    path('logout/', views.logout, name='logout'),
    path('course/', views.course, name='course'),
    path('course/learning', views.learning, name='learning'),
    path('course/learning/isl', views.isl, name='isl'),
    path('predict_hand_sign/', views.predict_hand_sign, name='predict_hand_sign'),
    
]