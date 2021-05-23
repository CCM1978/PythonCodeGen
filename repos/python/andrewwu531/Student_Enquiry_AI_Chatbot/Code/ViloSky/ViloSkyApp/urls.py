''' URLs for the ViloSky project '''
from django.urls import path, re_path
from ViloSkyApp import views
from django.contrib.auth.views import PasswordResetView, PasswordResetDoneView, PasswordResetConfirmView, PasswordResetCompleteView

urlpatterns = [
    path('', views.index, name='index'),
    path('login/', views.login, name='login'),
    path('register/', views.register, name='register'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('mydetails/', views.mydetails, name='mydetails'),
    re_path('action/(?P<report_id>\w+)/$', views.action, name='action'),
    path('actions/', views.actions, name='actions'),
    re_path('report/(?P<report_id>\w+)/$',
            views.report_view, name='report_view'),
    path('report/create', views.report_create, name='report_create'),
    path('report/view', views.report_view_public, name='report_view_public'),
    path('reports/', views.reports, name='reports'),

    # Admin Inputs
    re_path('admin_input/(?P<admin_input_id>\d+)/$',
            views.AdminInputDetail.as_view(), name='admin_input'),
    re_path('dropdown_admin_input/update/(?P<admin_input_id>\d+)/$',
            views.DropdownAdminInputUpdate.as_view(), name='dropdown_admin_input_update'),
    re_path('checkbox_admin_input/update/(?P<admin_input_id>\d+)/$',
            views.CheckboxAdminInputUpdate.as_view(), name='checkbox_admin_input_update'),
    re_path('multiselect_admin_input/update/(?P<admin_input_id>\d+)/$',
            views.MultiselectAdminInputUpdate.as_view(), name='multiselect_admin_input_update'),
    re_path('text_admin_input/update/(?P<admin_input_id>\d+)/$',
            views.TextAdminInputUpdate.as_view(), name='text_admin_input_update'),
    re_path('textarea_admin_input/update/(?P<admin_input_id>\d+)/$',
            views.TextareaAdminInputUpdate.as_view(), name='textarea_admin_input_update'),
    re_path('admin_input/delete/(?P<admin_input_id>\d+)/$',
            views.AdminInputDelete.as_view(), name='admin_input_delete'),
    path('admin_inputs/', views.admin_inputs, name='admin_inputs'),
    path('dropdown_admin_input/create/', views.DropdownAdminInput.as_view(),
         name='dropdown_admin_input_create'),
    path('checkbox_admin_input/create/', views.CheckboxAdminInput.as_view(),
         name='checkbox_admin_input_create'),
    path('multiselect_admin_input/create/', views.MultiselectAdminInput.as_view(),
         name='multiselect_admin_input_create'),
    path('text_admin_input/create/', views.TextAdminInput.as_view(),
         name='text_admin_input_create'),
    path('textarea_admin_input/create/', views.TextAreaAdminInput.as_view(),
         name='textarea_admin_input_create'),

    re_path('paragraph/(?P<paragraph_id>\w+)/$',
            views.paragraph, name='paragraph'),
    path('paragraphs/', views.paragraphs, name='paragraphs'),
    path('logout/', views.user_logout, name='logout'),

    # Reset Password
    path('reset-password/', PasswordResetView.as_view(), name='password_reset'),
    path('reset-password/done', PasswordResetDoneView.as_view(),
         name='password_reset_done'),
    path('reset-password/confirm/<uidb64>[0-9A-Za-z]+)-<token>/',
         PasswordResetConfirmView.as_view(), name='password_reset_confirm'),
    path('reset-password/complete/', PasswordResetCompleteView.as_view(),
         name='password_reset_complete'),
    path('create_paragraphs/', views.create_paragraphs, name='create_paragraphs'),
    path('create_pdf/(?P<report_id>\w+)/$',
         views.create_pdf, name='create_pdf'),
    path('create_pdf_public', views.create_pdf_public, name='create_pdf_public'),
]
