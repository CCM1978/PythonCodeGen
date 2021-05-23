from django.test import TestCase
from ViloSkyApp.forms import NewParaForm, NewKeywordForm, NewActionForm, NewKeywordForm, InputForm, UserProfileForm
import datetime
from django.utils import timezone

class CreateParaFormTests(TestCase):
    def test_accepts_just_para(self):
        newParaForm = NewParaForm({"static_text":"test adding paragraph"})
        self.assertTrue(newParaForm.is_valid())

class CreateKeywordFormTests(TestCase):
    def test_accepts_Keyword(self):
        newKeywordForm = NewKeywordForm({"url":"https://www.google.co.uk/"})
        self.assertTrue(newKeywordForm.is_valid())
    

class CreateActionFormTests(TestCase):
    def test_accepts_actions(self):
        newActionForm = NewActionForm({"title":"Check if we action can be added"})
        self.assertTrue(newActionForm.is_valid())

class CreateKeywordFormTests(TestCase):
    def test_accepts_input(self):
        newKeywordForm = NewKeywordForm({"key":"Test", "score":"10"})
        self.assertTrue(newKeywordForm.is_valid())
    
    def check_score_required(TestCase):
        newKeywordForm = NewKeywordForm({"key":"Test"})
        self.assertEqual(newKeywordForm.errors, {'score': ['Enter a valid score']})

    def check_score_required(TestCase):
        newKeywordForm = NewKeywordForm({"score":"1"})
        self.assertEqual(newKeywordForm.errors, {
            'score': ['Enter a valid key']})

class UserProfileFormTest(TestCase):
    def test_date_of_birth_field_label(self):
        form = UserProfileForm()
        self.assertTrue(form.fields['date_of_birth'].label == None or form.fields['date_of_birth'].label == 'date_of_birth')

