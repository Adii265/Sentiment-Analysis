from django.shortcuts import get_object_or_404,render
from django.http import HttpResponse
from .models import *
from django.template import loader
from django.urls import reverse
from django.views import generic
from Historie.forms import Doc_Form
from Historie.prog1 import cal


def index(request):
	return fill_doc(request)

def SampleView(request):
    model = Document
    latest_article_list = Document.objects.all()
    #print(latest_article_list)
    return render(request, 'Historie/sample.html', {'latest_article_list': latest_article_list})

def AboutView(request):
    #model = Document
    #template_name = 'Historie/about.html'
    return render(request,'Historie/about.html',)
def ContactView(request):
    #model = Document
    #template_name = 'Historie/about.html'
    return render(request,'Historie/contact.html',)

def fill_doc(request):
	sentiment = " "
	content = " "
	if request.method == "POST":
		My_Doc_Form = Doc_Form(request.POST)
		if My_Doc_Form.is_valid():
			author = My_Doc_Form.cleaned_data['author']
			content = My_Doc_Form.cleaned_data['content']
			sentiment = cal.c(author,content)
			p = Document(author = author,
						content = content,
						sentiment = sentiment)
			p.save()
	else:
		My_Doc_Form = Doc_Form()
	
	return render(request, 'Historie/index.html',{"content":content,"sentiment":sentiment })

def home(request):
	return render(request,'Historie/Home.html',)