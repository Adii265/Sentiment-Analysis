from django import forms

class Doc_Form(forms.Form):
	author = forms.CharField(max_length = 100)
	content = forms.CharField(max_length = 10000)
