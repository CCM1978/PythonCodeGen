from django import forms
from .models import Query

class QueryForm(forms.ModelForm):
    input = forms.CharField(
        label='',
        widget= forms.TextInput(
            attrs={
                'class' : 'msger-input',
                'autocomplete': 'off',
                'placeholder' : 'Type Something'
            }
        )
    )
    class Meta:
        model = Query
        fields = [
            'input',
        ]

