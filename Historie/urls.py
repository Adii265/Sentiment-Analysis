from django.urls import path
from . import views



urlpatterns = [
    path('', views.home, name='home'),
    path('index/', views.index, name='index'),
    path('sample/', views.SampleView, name='sample'),
    path('about/', views.AboutView, name='about'),
    path('contact/', views.ContactView, name='contact'),
    #path('<int:question_id>/vote/', views.vote, name='vote'),
]