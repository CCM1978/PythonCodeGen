'''Forms for the ViloSky app'''
import datetime
import json
from django import forms
from django.contrib.auth import get_user_model
from ViloSkyApp.models import (UserProfile, Qualification, AdminInput, DropdownAdminInput,
                               TextAdminInput, TextareaAdminInput, CheckboxAdminInput,
                               MultiselectAdminInput, Paragraph, Link, Keyword, Action)


class UserForm(forms.ModelForm):
    email = forms.CharField(widget=forms.EmailInput(
        attrs={'class': 'form-control form-control-lg',
               'id': 'inputEmail3', 'placeholder': 'Enter email'}))
    password = forms.CharField(widget=forms.PasswordInput(
        attrs={'class': 'form-control form-control-lg',
               'id': 'inputPassword4', 'placeholder': 'Enter password'}))
    confirm_password = forms.CharField(
        widget=forms.PasswordInput(
            attrs={'class': 'form-control form-control-lg',
                   'id': 'inputPassword5', 'placeholder': 'Confirm password'}))

    class Meta:
        model = get_user_model()
        fields = ('email', 'password', 'confirm_password')


class InputForm(forms.Form):
    def __init__(self, *args, **kwargs):
        super(InputForm, self).__init__(*args, **kwargs)
        for i, q in enumerate(AdminInput.objects.all()):
            # dropdowns
            if q.input_type == 'DROPDOWN':
                widget = forms.Select(attrs={"class": "form-control"})
                dropdown_q = DropdownAdminInput.objects.filter(
                    id=q.dropdownadmininput.id).first()
                if dropdown_q is not None:
                    # Need to set the choices as a tuple of (choice, choice)
                    # That way filling in the form with partial_inputs is possible
                    choices = [(c, c) for c in json.loads(dropdown_q.choices)]
                    # Set the default value of dropdowns to N/A
                    choices.insert(0, ('', "N/A"))
                    if q.is_required:
                        self.fields[dropdown_q.label] = forms.ChoiceField(
                            label=dropdown_q.label, choices=choices, widget=widget, required=True)
                    else:
                        self.fields[dropdown_q.label] = forms.ChoiceField(
                            label=dropdown_q.label, choices=choices, widget=widget, required=False)

            # Multiselect
            elif q.input_type == 'MULTISELECT':
                widget = forms.SelectMultiple(attrs={"class": "form-control"})
                multi_q = MultiselectAdminInput.objects.filter(
                    id=q.multiselectadmininput.id).first()
                if multi_q is not None:
                    # Need to set the choices as a tuple of (choice, choice)
                    # That way filling in the form with partial_inputs is possible
                    choices = [(c, c) for c in json.loads(multi_q.choices)]
                    if q.is_required:
                        self.fields[multi_q.label] = forms.MultipleChoiceField(
                            widget=widget, label=multi_q.label, choices=choices, required=True)
                    else:
                        self.fields[multi_q.label] = forms.MultipleChoiceField(
                            widget=widget, label=multi_q.label, choices=choices, required=False)
            # Checkboxes
            elif q.input_type == 'CHECKBOX':
                widget = forms.CheckboxInput()
                check_q = CheckboxAdminInput.objects.filter(
                    id=q.checkboxadmininput.id).first()
                if check_q is not None:
                    if q.is_required:
                        self.fields[check_q.label] = forms.BooleanField(
                            label=check_q.label, widget=widget, required=True)
                    else:
                        self.fields[check_q.label] = forms.BooleanField(
                            label=check_q.label, widget=widget, required=False)
            # Text
            elif q.input_type == 'TEXT':
                widget = forms.TextInput(attrs={"class": "form-control"})
                text_q = TextAdminInput.objects.filter(
                    id=q.textadmininput.id).first()
                if text_q is not None:
                    if q.is_required:
                        self.fields[text_q.label] = forms.CharField(
                            max_length=text_q.max_length, label=q.label,
                            widget=widget, required=True)
                    else:
                        self.fields[text_q.label] = forms.CharField(
                            max_length=text_q.max_length, label=q.label,
                            widget=widget, required=False)
            # TextAreas
            else:
                textarea_q = TextareaAdminInput.objects.filter(
                    id=q.textareaadmininput.id).first()
                if textarea_q is not None:
                    if q.is_required:
                        self.fields[textarea_q.label] = forms.CharField(
                            max_length=textarea_q.max_length, label=q.label, required=True)
                    else:
                        self.fields[textarea_q.label] = forms.CharField(
                            max_length=textarea_q.max_length, label=q.label, required=False)


class UserProfileForm(forms.ModelForm):
    cur_year = datetime.datetime.today().year
    date_of_birth = forms.DateField(widget=forms.SelectDateWidget(
        years=tuple(range(cur_year - 80, cur_year - 16)),
        attrs={
            'class': 'form-control snps-inline-select my-1 mr-1',
        }))

    class Meta:
        model = UserProfile
        fields = (
            'date_of_birth',
            'company',
            'employment_sector',
            'employment_status',
            'time_worked_in_industry'
        )
        widgets = {
            'company': forms.TextInput(attrs={'class': 'form-control',
                                              'placeholder': 'eg: ViloSky'}),
            'employment_sector': forms.TextInput(attrs={'class': 'form-control',
                                                        'placeholder': 'eg: Consulting'}),
            'employment_status': forms.Select(attrs={'class': 'form-control'}),
            'time_worked_in_industry': forms.Select(attrs={'class': 'form-control'}),
        }


class QualificationForm(forms.ModelForm):
    level = forms.CharField(max_length=160, widget=forms.TextInput(attrs={
        'placeholder': 'e.g. High School',
        'class': 'form-control'
    }),
        required=True)
    subjects = forms.CharField(max_length=150, widget=forms.TextInput(attrs={
        'placeholder': 'e.g. Maths',
        'class': 'form-control',
    }),
        required=True)

    class Meta:
        model = Qualification
        fields = (
            'level',
            'subjects'
        )


class DropdownAdminInputForm(forms.ModelForm):
    ''' Describes the form used to create a Dropdown AdminInput'''
    choices = forms.CharField(widget=forms.Textarea(
        attrs={'class': 'form-control form-control-lg',
               'rows': '3', 'placeholder': "Comma-separated values"}))

    class Meta:
        ''' Describes which fields to display and widgets regarding styling'''
        model = DropdownAdminInput
        fields = [
            'is_required',
            'label',
            'choices',
        ]
        widgets = {
            'label': forms.TextInput(attrs={'class': 'form-control form-control-lg',
                                            'placeholder': 'Enter input label'}),
        }


class CheckboxAdminInputForm(forms.ModelForm):
    ''' Describes the form used to create a Checkbox AdminInput'''

    class Meta:
        ''' Describes which fields to display and widgets regarding styling'''
        model = CheckboxAdminInput
        fields = [
            'is_required',
            'default_value',
            'label',
        ]
        widgets = {
            'label': forms.TextInput(attrs={'class': 'form-control form-control-lg',
                                            'placeholder': 'Enter input label'}),
        }


class MultiselectAdminInputForm(forms.ModelForm):
    ''' Describes the form used to create a Multiselect AdminInput'''
    choices = forms.CharField(widget=forms.Textarea(
        attrs={'class': 'form-control form-control-lg',
               'rows': '3', 'placeholder': "Comma-separated values"}))

    class Meta:
        ''' Describes which fields to display and widgets regarding styling'''
        model = MultiselectAdminInput
        fields = [
            'is_required',
            'label',
            'choices',
        ]
        widgets = {
            'label': forms.TextInput(attrs={'class': 'form-control form-control-lg',
                                            'placeholder': 'Enter input label'}),
        }


class TextAdminInputForm(forms.ModelForm):
    ''' Describes the form used to create a Text AdminInput'''

    class Meta:
        ''' Describes which fields to display and widgets regarding styling'''
        model = TextAdminInput
        fields = [
            'is_required',
            'label',
            'max_length',
        ]
        widgets = {
            'label': forms.TextInput(attrs={'class': 'form-control form-control-lg',
                                            'placeholder': 'Enter input label'}),
            'max_length': forms.NumberInput(attrs={'class': 'form-control form-control-lg',
                                                   'placeholder': 'Enter max length'}),
        }


class TextareaAdminInputForm(forms.ModelForm):
    ''' Describes the form used to create a TextArea AdminInput'''

    class Meta:
        ''' Describes which fields to display and widgets regarding styling'''
        model = TextareaAdminInput
        fields = [
            'is_required',
            'max_length',
            'label',
        ]
        widgets = {
            'label': forms.TextInput(attrs={'class': 'form-control form-control-lg',
                                            'placeholder': 'Enter input label'}),
            'max_length': forms.NumberInput(attrs={'class': 'form-control form-control-lg',
                                                   'placeholder': 'Enter max length'}),
        }


class NewParaForm(forms.ModelForm):
    class Meta:
        model = Paragraph
        fields = ('static_text',)
        widgets = {
            'static_text':  forms.Textarea(attrs={
                'placeholder': 'eg: Having been out of work for over a year...',
                'rows': '3', 'class': 'form-control'})}


class NewLinkForm(forms.ModelForm):
    class Meta:
        model = Link
        fields = ('url',)
        widgets = {
            'url': forms.TextInput(attrs={
                'placeholder': 'eg: https://womenreturners.com',
                'class': 'form-control'})}

    def __init__(self, *args, **kwargs):
        super(NewLinkForm, self).__init__(*args, **kwargs)
        self.fields['url'].required = False


class NewActionForm(forms.ModelForm):
    class Meta:
        model = Action
        fields = ('title',)
        widgets = {
            'title': forms.TextInput(attrs={
                'placeholder': 'eg: List your skills...',
                'class': 'form-control'})}

    def __init__(self, *args, **kwargs):
        super(NewActionForm, self).__init__(*args, **kwargs)
        self.fields['title'].required = False


class NewKeywordForm(forms.ModelForm):
    class Meta:
        model = Keyword
        fields = ('key', 'score')
        widgets = {
            'key': forms.TextInput(attrs={
                'placeholder': 'eg: Risk Management',
                'class': 'form-control mb-2'}),
            'score': forms.TextInput(attrs={'placeholder': 'eg: 10', 'class': 'form-control'})
        }

    def __init__(self, *args, **kwargs):
        super(NewKeywordForm, self).__init__(*args, **kwargs)
        self.fields['key'].required = False
        self.fields['score'].required = False
