from django.test import TestCase
from ..models import *
from django.contrib.auth import get_user_model
import datetime
import pytz


class TestUserModel(TestCase):
    @classmethod
    def setUpTestData(cls):
        # Set up non-modified objects used by all test methods
        User = get_user_model()
        User.objects.create(email="suzieMul23@gmail.com", first_name="Suzie", last_name="Mulligan").set_password(
            'testPassword123')

    def test_email_label(self):
        User = get_user_model()
        user = User.objects.get(id=1)
        field_label = user._meta.get_field('email').verbose_name
        self.assertEqual(field_label, 'email address')

    def test_first_name_label(self):
        User = get_user_model()
        user = User.objects.get(id=1)
        field_label = user._meta.get_field('first_name').verbose_name
        self.assertEqual(field_label, 'first name')

    def test_last_name_label(self):
        User = get_user_model()
        user = User.objects.get(id=1)
        field_label = user._meta.get_field('last_name').verbose_name
        self.assertEqual(field_label, 'last name')

    def test_first_name_max_length(self):
        User = get_user_model()
        user = User.objects.get(id=1)
        max_length = user._meta.get_field('first_name').max_length
        self.assertEqual(max_length, 150)


class TestUserProfileModel(TestCase):
    @classmethod
    def setUpTestData(cls):
        # Set up non-modified objects used by all test methods
        User = get_user_model()
        User.objects.create(email="suzieMul23@gmail.com", first_name="Suzie", last_name="Mulligan").set_password(
            'testPassword123')

        UserProfile.objects.create(user=User.objects.get(id=1),
                                   date_of_birth=datetime.datetime(1995, 11, 2, tzinfo=pytz.UTC),
                                   is_vilosky_admin=True,
                                   is_hr_representative=False,
                                   company='ViloSky',
                                   employment_sector='Consulting',
                                   employment_status='Employed',
                                   time_worked_in_industry=UserProfile.TimeWorkedTypes.ONE_TO_TWO_YEARS
                                   )

    def test_date_of_birth_label(self):
        user = UserProfile.objects.get(id=1)
        field_label = user._meta.get_field('date_of_birth').verbose_name
        self.assertEqual(field_label, 'date of birth')

    def test_is_vilosky_admin_label(self):
        user = UserProfile.objects.get(id=1)
        field_label = user._meta.get_field('is_vilosky_admin').verbose_name
        self.assertEqual(field_label, 'is vilosky admin')

    def test_is_hr_representative_label(self):
        user = UserProfile.objects.get(id=1)
        field_label = user._meta.get_field('is_hr_representative').verbose_name
        self.assertEqual(field_label, 'is hr representative')

    def test_company_label(self):
        user = UserProfile.objects.get(id=1)
        field_label = user._meta.get_field('company').verbose_name
        self.assertEqual(field_label, 'company')

    def test_company_max_length(self):
        user = UserProfile.objects.get(id=1)
        max_length = user._meta.get_field('company').max_length
        self.assertEqual(max_length, 255)

    def test_employment_sector_label(self):
        user = UserProfile.objects.get(id=1)
        field_label = user._meta.get_field('employment_sector').verbose_name
        self.assertEqual(field_label, 'employment sector')

    def test_employment_sector_max_length(self):
        user = UserProfile.objects.get(id=1)
        max_length = user._meta.get_field('employment_sector').max_length
        self.assertEqual(max_length, 255)

    def test_employment_status_label(self):
        user = UserProfile.objects.get(id=1)
        field_label = user._meta.get_field('employment_status').verbose_name
        self.assertEqual(field_label, 'employment status')

    def test_employment_status_max_length(self):
        user = UserProfile.objects.get(id=1)
        max_length = user._meta.get_field('employment_status').max_length
        self.assertEqual(max_length, 255)

    def test_time_worked_in_industry_label(self):
        user = UserProfile.objects.get(id=1)
        field_label = user._meta.get_field('time_worked_in_industry').verbose_name
        self.assertEqual(field_label, 'time worked in industry')
