from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth import get_user_model
import datetime
import pytz
from ViloSkyApp import models
from ..forms import *
import json
TEST_USER_PROFILE_PASSWORD = "MySecurePassword"

def create_test_user_profile():
    u = models.CustomUser.objects.get_or_create(email="a@a.com")[0]
    u.set_password(TEST_USER_PROFILE_PASSWORD)
    u.save()
    user = models.UserProfile.objects.get_or_create(user=u, company="aa")[0]
    user.save()
    return user

def create_test_qualification():
    ql = models.Qualification.objects.get_or_create(user=create_test_user_profile(),
                                                    level='ajbsncoqenfounsdfnw', subjects='jknqnefnojdnaxcnwenfi')[0]
    ql.save()
    return ql

def create_report_used_in_action():
    report = models.Report.objects.get_or_create(user=create_test_user_profile(), datetime_created=
                                                datetime.datetime.now())[0]
    report.save() 
    return report

def create_test_user_action():
    user_action = models.UserAction.objects.get_or_create(report=create_report_used_in_action(),
                                                           title = "This is a test action")[0]
    user_action.save()
    return user_action

class ActionPlanTests(TestCase):

    def test_no_listed_plans_on_plain_user(self):
        client = Client()
        user_profile = create_test_user_profile()
        client.login(username=user_profile.user.email,
                     password=TEST_USER_PROFILE_PASSWORD)
        response = client.get(reverse('actions'))
        self.assertContains(response, "No action plans found")
    
    def test_ordered_action_plans(self):
        client = Client()
        user_action = create_test_user_action()
        report = user_action.report
        user_profile = report.user
        report_past = models.Report.objects.create(user=user_profile, datetime_created = datetime.datetime.now() 
                                                            - datetime.timedelta(days=1))
        client.login(username=user_profile.user.email,
                     password=TEST_USER_PROFILE_PASSWORD)
        response = client.get(reverse('actions'))
        self.assertEqual(response.context['entries'][0].get("id"), report.id)

    def test_no_actions_report(self):
        client = Client()
        report = create_report_used_in_action()
        user_profile = report.user
        client.login(username=user_profile.user.email,
                     password=TEST_USER_PROFILE_PASSWORD)
        response = client.get('/action/1/') 
        self.assertContains(response, "No actions for this report")

    def test_display_actions(self):
        client = Client()
        action = create_test_user_action()
        report = action.report
        user_profile = report.user
        client.login(username=user_profile.user.email,
                     password=TEST_USER_PROFILE_PASSWORD)
        response = client.get('/action/1/') 
        self.assertContains(response, "This is a test action") 

    def test_update_action_plan(self):
        client = Client()
        action = create_test_user_action()
        report = action.report
        user_profile = report.user
        action2 = models.UserAction.objects.create(report=report, 
                                                            title="a", is_completed=True)
        client.login(username=user_profile.user.email,
                     password=TEST_USER_PROFILE_PASSWORD)
        response = client.post('/action/1/', {'completed': [action.id], 'not_completed': [action2.id]})
        action = models.UserAction.objects.get(id=action.id)
        action_from_completed_to_uncompleted = models.UserAction.objects.get(id=action2.id) 
        self.assertEqual(action.is_completed, True)
        self.assertEqual(action_from_completed_to_uncompleted.is_completed, False)

    def test_failed_post(self):
        client = Client()
        action = create_test_user_action()
        report = action.report
        user_profile = report.user
        client.login(username=user_profile.user.email,
                     password=TEST_USER_PROFILE_PASSWORD)
        response = client.post('/action/1/', {'not_completed': ''})
        self.assertEqual(response.context['success'], False)

        
class Profile_page_testing(TestCase):

    def test_qualification_displayed(self):
        client = Client()
        ql = create_test_qualification()
        user_profile = create_test_user_profile()
        client.login(username=user_profile.user.email,
                     password=TEST_USER_PROFILE_PASSWORD) 
        response = client.get(reverse('mydetails'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, ql.level)
        self.assertContains(response, ql.subjects) 

    #test qualification can be deleted
    def test_delete_qualification(self):
        client = Client()
        ql = create_test_qualification()
        user_profile = ql.user
        client.login(username=user_profile.user.email,
                     password=TEST_USER_PROFILE_PASSWORD)
        response = client.post(reverse('mydetails'), {'delete_list' : '1', 'delete_qualifications' : ''})
        self.assertIsNone(response.context)

    # test new qualification added successfully
    def test_add_qual(self):
        client = Client()
        user_profile = create_test_user_profile()
        client.login(username=user_profile.user.email,
                     password=TEST_USER_PROFILE_PASSWORD) 
        client.post(reverse('mydetails'), {'level' : 'QWERTYUI', 'subjects' : 'Law', 'addquals' : ''})
        ql_added = models.Qualification.objects.get(user = user_profile)
        self.assertEqual(ql_added.user, user_profile)
        self.assertEqual(ql_added.level, 'QWERTYUI')
        self.assertEqual(ql_added.subjects, 'Law')

    #test invalid qualification form posted
    def test_add_qual_failed(self):
        client = Client()
        user_profile = create_test_user_profile()
        client.login(username=user_profile.user.email,
                     password=TEST_USER_PROFILE_PASSWORD) 
        response = client.post(reverse('mydetails'), {'level' : 'ajbsncoqenfounsdfnw', 'subjects' : '', 'addquals' : ''})
        qls_length = len(models.Qualification.objects.all())
        self.assertEqual(qls_length, 0)


    def test_edit_profile(self):
        client = Client()
        user_profile = create_test_user_profile()
        client.login(username=user_profile.user.email,
                     password=TEST_USER_PROFILE_PASSWORD) 
        response = client.post(reverse('mydetails'), {'date_of_birth_month': ['10'], 'date_of_birth_day': ['5'], 
                                                        'date_of_birth_year': ['1980'], 'company': ['haha'], 
                                                        'employment_status': [''], 'employment_sector': [''],
                                                        'time_worked_in_industry': ['']})
        profile = models.UserProfile.objects.get(company="haha")
        self.assertEqual(profile, user_profile)


class test_register_view(TestCase):
    @classmethod
    def setUp(self):
        User = get_user_model()
        User.objects.create(email="apple@apple.com", first_name="App", last_name="Lee").set_password(
            'testPassword123')
        
    
    def test_register_correct_template(self):
        response = self.client.get(reverse('register'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'register.html')
    
    def test_register_registers_a_user(self):
        User = get_user_model()
        user = User.objects.get(id=1)
        user_details = {"email":"apple@apple.com", "password":"applepie"}
        response = self.client.post(reverse('register'), data=user_details)
        self.assertTrue(User.objects.filter(email= 'apple@apple.com').exists())


class test_dashboard(TestCase):
    @classmethod
    def setUp(self):

        User = get_user_model()
        self.client = Client()

        u1 = User.objects.get_or_create(email="test@me.com")[0]
        u1.set_password("abcd1234")
        u1.save()
        user1 = UserProfile.objects.get_or_create(user=u1)[0]
        user1.save()

        u2 = User.objects.get_or_create(email="testB@me.com")[0]
        u2.set_password("abcd1234")
        u2.save()
        user2 = UserProfile.objects.get_or_create(user=u2)[0]
        user2.save()

        # Create a report instance and add paragraphs to it
        self.rep1 = models.Report(user=user1,
                        datetime_created=datetime.datetime(2021, 11, 2, 20, 8, 7, 0, tzinfo=pytz.UTC))
        self.rep1.save()
        self.rep2 = models.Report(user=user1,
                        datetime_created=datetime.datetime(2021, 11, 3, 20, 8, 7, 0, tzinfo=pytz.UTC))
        self.rep2.save()
        self.rep3 = models.Report(user=user1,
                        datetime_created=datetime.datetime(2021, 11, 1, 20, 8, 7, 0, tzinfo=pytz.UTC))
        self.rep3.save()

        self.response = self.client.get('/dashboard/')

    def test_register_correct_template(self):
        self.assertEqual(self.response.status_code, 302)
        str_url = str(self.response)
        self.assertTrue(str_url.__contains__("dashboard"))

    def test_correct_rep_shown(self):
        User = get_user_model()
        user = User.objects.get(email="test@me.com").user_profile
        self.client.login(email="test@me.com", password="abcd1234")
        
        recent = user.reports_assigned.all().order_by('-datetime_created').first()
        self.assertEqual(recent, self.rep2)

    def test_no_rep_shown(self):
        User = get_user_model()
        user = User.objects.get(email="testB@me.com").user_profile
        self.client.login(email="test@me.com", password="abcd1234")
        
        recent = user.reports_assigned.all().order_by('-datetime_created').first()
        self.assertIsNone(recent)
        
    def test_wrong_rep_shown(self):
        User = get_user_model()
        user = User.objects.get(email="test@me.com").user_profile
        self.client.login(email="test@me.com", password="abcd1234")
        
        recent = user.reports_assigned.all().order_by('-datetime_created').first()
        self.assertNotEqual(recent, self.rep1)
        self.assertNotEqual(recent, self.rep3)

class CURDAdminInputsTest(TestCase):
    def setUp(self):
        User = get_user_model()
        u1 = User.objects.get_or_create(email="test@me.com")[0]
        u1.set_password("abcd1234")
        u1.save()
        user1 = UserProfile.objects.get_or_create(user=u1)[0]
        user1.save()
        u1.is_superuser = True
        u1.save()

        AdminInput.objects.get_or_create(created_by=user1, label="Name", input_type="TEXT",
                                        is_required=False, id=1)
        admin_input = AdminInput.objects.get(label="Name")
        a1 = TextAdminInput.objects.create(created_by=user1, admin_input=admin_input, max_length=24)
        a1.save()
        admin_input.save()

    # Check if a logged in admin user can view the list of admin_inputs and the correct template is user
    def test_admin_input_list(self):
        #Log in with an admin user
        self.client.login(email="test@me.com", password="abcd1234")
        response = self.client.get(reverse('admin_inputs'))

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed('admin_inputs.html')

    # Check if a anonymous user(not logged in) gets redirected away from the list of admin_inputs
    def test_admin_input_list_anonymous(self):
        response = self.client.get(reverse('admin_inputs'))

        self.assertEqual(response.status_code, 302)

    # Check if admin user can access an admin_input_detail page
    def test_admin_input_detail(self):
        self.client.login(email="test@me.com", password="abcd1234")
        response = self.client.get(reverse('admin_input', kwargs={'admin_input_id': 1}))
        self.assertEqual(response.status_code, 200)

    # Check if anonymous user cannot access an admin_input_detail page. And is redirected away.
    def test_admin_input_detail_anonymous(self):
        response = self.client.get(reverse('admin_input', kwargs={'admin_input_id': 1}))
        self.assertEqual(response.status_code, 302)

    # Check if a logged in user can add admin inputs
    def test_admin_input_add(self):
        self.client.login(email="test@me.com", password="abcd1234")
        response = self.client.post(reverse('text_admin_input_create'),{
            'label': "Test",
            'max_length': 24,
            'is_required': True
        })

        # Get the created admin input and check if it's there.
        self.assertEqual(response.status_code, 302)
        self.assertEqual(TextAdminInput.objects.get(label="Test").label, "Test")

    # Check if an anonymous user cannot add admin inputs
    def test_admin_input_add_anonymous(self):
        response = self.client.post(reverse('text_admin_input_create'), {
            'label': "Test",
            'max_length': 24,
            'is_required': True
        })

        # Get the created admin input and check that it wasn't created and the user was redirected
        self.assertEqual(response.status_code, 302)
        self.assertEqual(TextAdminInput.objects.filter(label="Test").first(), None)

    # Check if an admin user can update admin inputs
    def test_admin_input_update(self):
        self.client.login(email="test@me.com", password="abcd1234")
        response = self.client.post(reverse('text_admin_input_update', kwargs={'admin_input_id': 1}), {
            'label': 'Surname'
        })
        # Assert that the label updated
        self.assertEqual(TextAdminInput.objects.get(id=1).label, "Surname")

    # Check if an anonymous user cannot update admin inputs
    def test_admin_input_update_anonymous(self):
        response = self.client.post(reverse('text_admin_input_update', kwargs={'admin_input_id': 1}), {
            'label': 'Surname'
        })

        # Assert that the label didn't update
        self.assertEqual(response.status_code, 302)
        self.assertEqual(TextAdminInput.objects.get(id=1).label, 'Name')

    # Check if admin user can delete admin inputs
    def test_admin_input_delete(self):
        self.client.login(email="test@me.com", password="abcd1234")
        self.client.post(reverse('admin_input_delete', kwargs={'admin_input_id': 1}))

        # Check if object was removed.
        self.assertEqual(TextAdminInput.objects.filter(label="Name").first(), None)

    # Check if anonymous user cannot delete admin inputs
    def test_admin_input_delete_anonymous(self):
        self.client.post(reverse('admin_input_delete', kwargs={'admin_input_id': 1}))

        # assert that object was not removed.
        self.assertEqual(TextAdminInput.objects.get(id=1).label, 'Name')

class test_create_para_view(TestCase):
    @classmethod
    def setUp(self):
         newParaForm = NewParaForm({"static_text":"test adding paragraph"})
    def correct_response(self):
        response = self.client.get(reverse('paragraphs'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'paragraphs.html')

