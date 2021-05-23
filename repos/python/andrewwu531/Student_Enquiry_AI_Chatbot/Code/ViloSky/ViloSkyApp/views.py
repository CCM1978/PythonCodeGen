""" Views for the ViloSky app"""
import json
from random import uniform, randint
from datetime import datetime
from difflib import SequenceMatcher
from plotly.offline import plot
import plotly.graph_objs as go
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse
from django.contrib import messages
from django.contrib.auth import authenticate
from django.contrib.auth import login as ulogin
from django.contrib.auth import logout as ulogout
from django.contrib.auth.decorators import login_required
from django.core.serializers import serialize
from django.utils.formats import localize
from ViloSkyApp import models
from ViloSkyApp.forms import (UserForm, QualificationForm, UserProfileForm, DropdownAdminInputForm,
                              CheckboxAdminInputForm, TextAdminInputForm, TextareaAdminInputForm,
                              InputForm, NewParaForm, NewActionForm, NewLinkForm, NewKeywordForm,
                              MultiselectAdminInputForm)
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import CreateView, UpdateView, DeleteView, DetailView
from django.forms import modelformset_factory, formset_factory
from fpdf import FPDF
from django.http import FileResponse
import io


def index(request):
    return redirect(reverse('login'))


def login(request):
    context_dict = {}
    if not request.user.is_authenticated:
        if request.method == 'POST':
            email = request.POST.get('email')
            password = request.POST.get('password')
            user = authenticate(username=email, password=password)
            if user:
                ulogin(request, user)
                return redirect(reverse('dashboard'))
            else:
                messages.error(request, "Email or password is incorrect")
                return redirect(reverse('login'))
        else:
            return render(request, 'login.html', context_dict)
    else:
        return redirect(reverse('dashboard'))
    return render(request, 'login.html', {})


def register(request):
    if not request.user.is_authenticated:
        registered = False
        if request.method == 'POST':
            user_form = UserForm(request.POST)
            if user_form.is_valid():
                user = user_form.save()
                user.set_password(user.password)
                user.save()
                if request.POST.get('tos') != "on":
                    user.delete()
                    messages.error(
                        request, "You must accept the TOS and Privacy Policy in order to register.")
                elif request.POST.get('password') != request.POST.get('confirm_password'):
                    user.delete()
                    messages.error(
                        request, "The passwords provided do not match.")
                else:
                    registered = True

                if registered:
                    user_profile = models.UserProfile.objects.create(
                        user=user, date_of_birth=None, company="", employment_sector="", employment_status="", time_worked_in_industry="")
                    user_profile.save()
        else:
            user_form = UserForm()
        context_dict = {
            'user_form': user_form,
            'registered': registered,
        }
        return render(request, 'register.html', context_dict)
    else:
        return redirect(reverse('dashboard'))


@login_required(login_url='login')
def user_logout(request):
    ulogout(request)
    return redirect(reverse('index'))


@login_required(login_url='login')
def dashboard(request):
    user = request.user.user_profile
    report = user.reports_assigned.all().order_by('-datetime_created').first()
    paras = []
    links_dict = {}

    if report is not None:
        paras = report.paragraphs.all()
        link_list = models.Link.objects.filter(paragraph__in=paras)
        actions_list = models.Action.objects.filter(paragraph__in=paras)

        # get the associated links and actions for each paragraph
        for par in paras:
            temp = []
            t = []
            big_l = []
            for act in actions_list:
                if par == act.paragraph:
                    t.append(act)
            for link in link_list:
                if par == link.paragraph:
                    temp.append(link)
            if t:
                big_l.append(t)
            if temp:
                big_l.append(temp)
            # list of the lists for links and actions added to dictionary
            links_dict[par] = big_l

    actions = models.UserAction.objects.filter(report=report)

    return render(request, 'dashboard.html', {'report': report, 'actions': actions, 'paras': paras, 'data': links_dict, })


@login_required(login_url='login')
def mydetails(request):
    p_form = UserProfileForm(instance=request.user.user_profile)
    q_form = QualificationForm(request.POST)

    if request.method == 'POST':
        if 'addquals' in request.POST:
            if q_form.is_valid():
                quals = q_form.save(commit=False)
                quals.user = request.user.user_profile
                quals.save()
                return redirect(reverse('mydetails'))
            else:
                q_form = QualificationForm(instance=request.user.user_profile)
        elif 'delete_qualifications' in request.POST:
            models.Qualification.objects.filter(
                pk__in=request.POST.getlist('delete_list')).delete()
            return redirect(reverse('mydetails'))
        else:
            p_form = UserProfileForm(
                request.POST, instance=request.user.user_profile)
            if p_form.is_valid():
                profile = p_form.save()
                profile.save()
                return redirect(reverse('mydetails'))
            else:
                p_form = UserProfileForm(instance=request.user.user_profile)
    qualifications = models.Qualification.objects.filter(
        user=request.user.user_profile)
    context_dict = {'p_form': p_form, 'q_form': q_form,
                    'qualifications': qualifications}
    return render(request, 'myDetails.html', context=context_dict)


@login_required(login_url='login')
def action(request, report_id):
    # Pass required info and update template
    actions_list = models.UserAction.objects.all().filter(report=report_id)

    if request.method == "GET":
        context_dict = {
            'actions': actions_list,
            'action_id': report_id,
        }
        return render(request, 'action_plan.html', context_dict)

    elif request.method == 'POST':
        is_success = False
        try:
            if 'completed' in request.POST:
                completed_actions_ids = request.POST.getlist(
                    'completed')
                completed_actions = actions_list.filter(
                    pk__in=completed_actions_ids)
                incomplete_actions = actions_list.exclude(
                    id__in=completed_actions_ids)
                completed_actions.update(is_completed=True)
                incomplete_actions.update(is_completed=False)
                is_success = True

            context_dict = {
                'actions': actions_list,
                'action_id': report_id,
                'success': is_success,
            }
            return render(request, 'action_plan.html', context_dict)
        except Exception as _:
            context_dict = {
                'actions': actions_list,
                'action_id': report_id,
                'success': False,
            }
            return render(request, 'action_plan.html', context_dict)


@login_required(login_url='login')
def actions(request):
    user = models.UserProfile.objects.filter(user=request.user).first()

    if user is None:
        return render(request, "error.html")

    user_reports = user.reports_assigned.all().order_by('-datetime_created')
    entries = [{"id": report.id, "title": f"Action plan from {localize(report.datetime_created)}"}
               for report in user_reports]
    template_headings = ["#", "Title"]
    model_keys = ["id", "title"]

    return render(request, 'action_plans.html', {
        "headings": template_headings, "model_keys": model_keys,
        "entries": entries, "row_link_to": "/action/"
    })


@login_required
def baseuser(request):
    return render(request, 'baseuser.html', {})


@login_required(login_url='login')
def reports(request):
    user = models.UserProfile.objects.filter(user=request.user).first()

    if user is None:
        return render(request, "error.html")

    reports_to_render = list(
        user.reports_assigned.all().values().order_by('-datetime_created'))
    for report in reports_to_render:
        report['datetime_created'] = localize(report['datetime_created'])
    template_headings = ["#", "Date Created"]
    model_keys = ["id", "datetime_created"]

    return render(request, 'reports.html', {
        "headings": template_headings, "model_keys": model_keys,
        "entries": reports_to_render, "row_link_to": "/report/"
    })


class AdminInputDetail(LoginRequiredMixin, DetailView):
    model = models.AdminInput
    template_name = 'admin_input_detail.html'
    # By default django looks for "pk" in the url
    # By changing this it will now look for admin_input_id instead, as is already written in urls.py
    pk_url_kwarg = 'admin_input_id'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        # If dropdown or multiselect, load choices properly for display.
        if context['object'].input_type == models.AdminInput.AdminInputTypes.DROPDOWN:
            context['choices'] = ", ".join(json.loads(
                context['object'].dropdownadmininput.choices))
        elif context['object'].input_type == models.AdminInput.AdminInputTypes.MULTISELECT:
            context['choices'] = ", ".join(json.loads(
                context['object'].multiselectadmininput.choices))
        return context


class DropdownAdminInput(LoginRequiredMixin, CreateView):
    form_class = DropdownAdminInputForm
    template_name = 'admin_input_form.html'

    def form_valid(self, form):
        form.instance.created_by = get_object_or_404(
            models.UserProfile, user=self.request.user)
        form.instance.input_type = "DROPDOWN"

        for ch in [' ', '[', ']', '\\', '\"']:
            if ch in form.instance.choices:
                form.instance.choices = form.instance.choices.replace(ch, '')
        form.instance.choices = json.dumps(form.instance.choices.split(','))

        return super().form_valid(form)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = "Create Dropdown Input"
        return context


class DropdownAdminInputUpdate(LoginRequiredMixin, UpdateView):
    form_class = DropdownAdminInputForm
    model = models.DropdownAdminInput
    template_name = 'admin_input_form.html'
    pk_url_kwarg = 'admin_input_id'

    def form_valid(self, form):
        for ch in [' ', '[', ']', '\\', '\"', '\'']:
            if ch in form.instance.choices:
                form.instance.choices = form.instance.choices.replace(ch, '')

        form.instance.choices = json.dumps(form.instance.choices.split(','))
        return super().form_valid(form)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = "Edit Dropdown Input"
        return context


class CheckboxAdminInput(LoginRequiredMixin, CreateView):
    form_class = CheckboxAdminInputForm
    template_name = 'admin_input_form.html'

    def form_valid(self, form):
        form.instance.created_by = get_object_or_404(
            models.UserProfile, user=self.request.user)
        form.instance.input_type = "CHECKBOX"
        return super().form_valid(form)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = "Create Checkbox Input"
        return context


class CheckboxAdminInputUpdate(LoginRequiredMixin, UpdateView):
    form_class = CheckboxAdminInputForm
    model = models.CheckboxAdminInput
    template_name = 'admin_input_form.html'
    pk_url_kwarg = 'admin_input_id'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = "Edit Checkbox Input"
        return context


class MultiselectAdminInput(LoginRequiredMixin, CreateView):
    form_class = MultiselectAdminInputForm
    template_name = 'admin_input_form.html'

    def form_valid(self, form):
        form.instance.created_by = get_object_or_404(
            models.UserProfile, user=self.request.user)
        form.instance.input_type = "MULTISELECT"

        for ch in [' ', '[', ']', '\\', '\"']:
            if ch in form.instance.choices:
                form.instance.choices = form.instance.choices.replace(ch, '')
        form.instance.choices = json.dumps(form.instance.choices.split(','))

        return super().form_valid(form)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = "Create Multiselect Input"
        return context


class MultiselectAdminInputUpdate(LoginRequiredMixin, UpdateView):
    form_class = MultiselectAdminInputForm
    model = models.MultiselectAdminInput
    template_name = 'admin_input_form.html'
    pk_url_kwarg = 'admin_input_id'

    def form_valid(self, form):
        for ch in [' ', '[', ']', '\\', '\"', '\'']:
            if ch in form.instance.choices:
                form.instance.choices = form.instance.choices.replace(ch, '')

        form.instance.choices = json.dumps(form.instance.choices.split(','))
        return super().form_valid(form)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = "Edit Multiselect Input"
        return context


class TextAdminInput(LoginRequiredMixin, CreateView):
    form_class = TextAdminInputForm
    template_name = 'admin_input_form.html'

    def form_valid(self, form):
        form.instance.created_by = get_object_or_404(
            models.UserProfile, user=self.request.user)
        form.instance.input_type = "TEXT"
        return super().form_valid(form)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = "Create Text Input"
        return context


class TextAdminInputUpdate(LoginRequiredMixin, UpdateView):
    form_class = TextAdminInputForm
    model = models.TextAdminInput
    template_name = 'admin_input_form.html'
    pk_url_kwarg = 'admin_input_id'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = "Edit Text Input"
        return context


class TextAreaAdminInput(LoginRequiredMixin, CreateView):
    form_class = TextareaAdminInputForm
    template_name = 'admin_input_form.html'

    def form_valid(self, form):
        form.instance.created_by = get_object_or_404(
            models.UserProfile, user=self.request.user)
        form.instance.input_type = "TEXTAREA"
        return super().form_valid(form)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = "Create Text Area Input"
        return context


class TextareaAdminInputUpdate(LoginRequiredMixin, UpdateView):
    form_class = TextareaAdminInputForm
    model = models.TextareaAdminInput
    template_name = 'admin_input_form.html'
    pk_url_kwarg = 'admin_input_id'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = "Edit Text Area Input"
        return context


class AdminInputDelete(LoginRequiredMixin, DeleteView):
    model = models.AdminInput
    template_name = 'admin_input_delete.html'
    pk_url_kwarg = 'admin_input_id'

    def get_success_url(self):
        return reverse('admin_inputs')


@ login_required(login_url='login')
def admin_inputs(request):
    # Get User
    user = models.UserProfile.objects.filter(user=request.user).first()

    # If no user logged in then show error.html
    if user is None:
        return render(request, "error.html")

    # List Item to store admin input details to be rendered
    inputs_to_render = list(
        models.AdminInput.objects.all().order_by("-id").values('id', 'created_by__user__first_name', 'label', 'input_type',
                                                               'is_required'))
    template_headings = ["#", "Created By", "Label", "Type", "Required"]
    model_keys = ["id", "created_by__user__first_name",
                  "label", "input_type", "is_required"]
    for entry in inputs_to_render:
        entry['input_type'] = entry['input_type'].capitalize()

    return render(request, 'admin_inputs.html', {
        "headings": template_headings, "model_keys": model_keys,
        "entries": inputs_to_render, "row_link_to": "/admin_input/"
    })


@ login_required(login_url='login')
def paragraph(request, paragraph_id):
    created_by = models.UserProfile.objects.filter(user=request.user).first()
    page = 'paragraph/' + paragraph_id + '/'
    # Pass required info and update template
    para = models.Paragraph.objects.filter(id=paragraph_id)[0]
    links = models.Link.objects.filter(paragraph=paragraph_id)
    actions = models.Action.objects.filter(paragraph=paragraph_id)
    keywords = models.Keyword.objects.filter(paragraph=paragraph_id)

    paragraph = models.Paragraph.objects.get(id=paragraph_id)

    # all these forms are to edit paragraphs
    p_form = NewParaForm(request.POST, instance=paragraph)
    link_form = NewLinkForm(request.POST)
    action_form = NewActionForm(request.POST)
    keywords_form = NewKeywordForm(request.POST)

    if request.method == 'POST':
        if 'editText' in request.POST:
            if p_form.is_valid():
                p = p_form.save(commit=False)
                p.created_by = created_by
                p.save()
                return redirect(reverse('paragraphs'))
        elif 'editLinks' in request.POST:
            if link_form.is_valid():
                l = link_form.save(commit=False)
                l.paragraph = models.Paragraph.objects.get(id=paragraph_id)
                l.save()
                return redirect(reverse('paragraphs'))
        elif 'editActions' in request.POST:
            if action_form.is_valid():
                a = action_form.save(commit=False)
                a.paragraph = models.Paragraph.objects.get(id=paragraph_id)
                a.save()
                return redirect(reverse('paragraphs'))
        elif 'editKeys' in request.POST:
            if keywords_form.is_valid():
                k = keywords_form.save(commit=False)
                k.paragraph = models.Paragraph.objects.get(id=paragraph_id)
                k.save()
                return redirect('paragraphs')
        elif 'delete_actions' in request.POST:
            models.Action.objects.filter(
                pk__in=request.POST.getlist('delete_list')).delete()
            return redirect(reverse('paragraphs'))
        elif 'delete_links' in request.POST:
            models.Link.objects.filter(
                pk__in=request.POST.getlist('delete_list')).delete()
            return redirect(reverse('paragraphs'))
        elif 'delete_keywords' in request.POST:
            models.Keyword.objects.filter(
                pk__in=request.POST.getlist('delete_list')).delete()
            return redirect(reverse('paragraphs'))
        else:
            models.Paragraph.objects.filter(id=paragraph_id).delete()
            return redirect(reverse('paragraphs'))

    return render(request, 'paragraph.html', {'paragraph_id': paragraph_id, 'created_by': created_by, 'para': para, 'links': links,
                                              'keywords': keywords, 'actions': actions, 'p_form': p_form, 'l_form': link_form, 'a_form': action_form,
                                              'k_form': keywords_form})


@ login_required(login_url='login')
def paragraphs(request):
    user = models.UserProfile.objects.filter(user=request.user).first()

    if user is None:
        return render(request, "error.html")

    pars_to_render = list(
        models.Paragraph.objects.all()
        .order_by('-id').values('id', 'created_by__user__first_name', 'static_text'))
    template_headings = ["#", "Created By", "Text"]
    model_keys = ["id", "created_by__user__first_name", "static_text"]

    return render(request, 'paragraphs.html', {
        "headings": template_headings, "model_keys": model_keys,
        "entries": pars_to_render, "row_link_to": "/paragraph/"
    })


# @ login_required(login_url='login')
# def data(request):
#     visitors = []
#     registered_users = []
#     inputs = []
#     outputs = []
#     days = []
#     for i in range(31):
#         days.append(i)
#         cur_visitors = randint(0, 12000)
#         visitors.append(cur_visitors)
#         registered_users.append(int(cur_visitors*uniform(0, 0.4)))
#         inputs.append(randint(100, 2900))
#         outputs.append(randint(200, 8000))
#     vis_fig = go.Figure(data=[go.Scatter(x=days, y=visitors, name="visitors"),
#                               go.Scatter(x=days, y=registered_users, name="registered users")]
#                         )
#     vis_div = plot(vis_fig, output_type='div')
#     inp_fig = go.Figure(data=go.Scatter(x=days, y=inputs, name="inputs"))
#     inp_div = plot(inp_fig, output_type='div')
#     out_fig = go.Figure(data=go.Scatter(x=days, y=outputs, name="outputs"))
#     out_div = plot(out_fig, output_type='div')
#     content_dict = {"vis_div": vis_div, "inp_div": inp_div, "out_div": out_div}
#     return render(request, 'data.html', content_dict)


def report_create(request):
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # Create a form instance and fill it with the request data
        input_form = InputForm(request.POST)

        if input_form.is_valid():
            # data from the form is stored in a dictionary, 'cd'
            cd = input_form.cleaned_data
            request.session['saved'] = cd
            # Create report
            report_id = create_report(request,
                                      cd, request.user.is_authenticated)
            if report_id is not None:
                return redirect(reverse('report_view', kwargs={'report_id': report_id}))
            else:
                return redirect(reverse('report_view_public'))
        else:
            context = {'inputForm': input_form}
            return render(request, 'report_create.html', context)

    # if we arrive here from a GET method, i.e. via inputting a url.
    else:
        # Create a new instance of the form
        input_form = InputForm()

        # Check if can populate the database from partial_inputs. i.e. if the user left without completing the form
        if request.user.is_authenticated:
            user = models.UserProfile.objects.get(user=request.user)
            partials = models.PartialInput.objects.filter(created_by=user)
            if partials:
                # Dictionary to store partials, to be passed in to instantiated input_form
                partials_dict = {}
                for p in partials:
                    if p.admin_input.input_type == "DROPDOWN":
                        partials_dict[p.admin_input.label] = (p.value, p.value)
                    elif p.admin_input.input_type == "MULTISELECT":
                        partials_dict[p.admin_input.label] = p.value.strip(
                            "[]").replace("\'", "").split(", ")
                    else:
                        partials_dict[p.admin_input.label] = p.value
                input_form = InputForm(initial=partials_dict)
    context = {'inputForm': input_form, }
    return render(request, 'report_create.html', context)


@login_required(login_url='login')
def report_view(request, report_id):
    # Pass required info and update template
    report = models.Report.objects.filter(id=int(report_id)).first()
    if report is not None:
        links_dict = compile_report(report)
        return render(request, 'report.html', {'data': links_dict, 'report_id': report_id,
                                               'datetime': report.datetime_created})
    return render(request, "error.html")


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def get_paragraphs(inputs_dictionary):
    if not inputs_dictionary:
        return None
    # Get all keywords, questions and list of answers
    keywords = models.Keyword.objects.all()
    question_list = models.AdminInput.objects.all()
    answers = [value for value in inputs_dictionary.values()]

    # create dictionary of paragraph score(how relevant paragraphs are to user input)
    paragraphs_list = []
    scores_dict = {}
    counter = 0

    for question in question_list:
        if question.input_type == 'CHECKBOX':
            if answers[counter] == 'True':
                counter += 1
        elif question.input_type == 'DROPDOWN':
            for keyword in keywords:
                if keyword == answers[counter]:
                    paragraphs_list.append(keyword.paragraph)
            counter += 1
        elif question.input_type == 'MULTISELECT':
            for keyword in keywords:
                if keyword.key in answers[counter]:
                    paragraphs_list.append(keyword.paragraph)
            counter += 1
        else:
            for keyword in keywords:
                score = similarity(answers[counter], keyword.key)
                para = keyword.paragraph
                if para not in scores_dict.keys():
                    scores_dict[para] = score
                else:
                    scores_dict[para] += score
            counter += 1

    num_paras = 5

    for _ in range(num_paras):
        highest_score = max(scores_dict, key=scores_dict.get)
        paragraphs_list.append(highest_score)
        del scores_dict[highest_score]
    return paragraphs_list


def compile_report(report):
    paras = report.paragraphs.all()
    link_list = models.Link.objects.filter(paragraph__in=paras)
    actions_list = models.Action.objects.filter(paragraph__in=paras)

    links_dict = {}

    # get the associated links and actions for each paragraph
    for par in paras:
        temp = []
        t = []
        big_l = []
        for link in link_list:
            if par == link.paragraph:
                temp.append(link)
        for act in actions_list:
            if par == act.paragraph:
                t.append(act)
        if temp:
            big_l.append(temp)
        if t:
            big_l.append(t)
        # list of the lists for links and actions added to dictionary
        links_dict[par] = big_l
    return links_dict


def create_pdf(request, report_id):
    # Pass required info and update template
    report = models.Report.objects.filter(id=int(report_id)).first()

    if report is not None:
        links_dict = compile_report(report)
        # now start constructing the pdf
        pdf_file = pdf_creator(links_dict)
        binaryIO = io.BytesIO(pdf_file)
        binaryIO.seek(0)
        return FileResponse(binaryIO, content_type='application/pdf', as_attachment=True, filename='report.pdf')

    return render(request, "error.html")


def report_view_public(request):
    # Get the dictionary of inputs. Gathered in InputForm, saved to django session.
    inputs = request.session.get('saved')
    # Get a list of paragraphs based on the inputs
    paras = get_paragraphs(inputs)
    if paras is None:
        return render(request, 'report_public.html', {'error': True})
    # Get all the relevant context for the paragraphs
    links_dict = get_context_from_paragraphs(paras)
    context = {'data': links_dict}
    return render(request, 'report_public.html', context)


def create_pdf_public(request):
    # Get the dictionary of inputs. Gathered in InputForm, saved to django session.
    inputs = request.session.get('saved')
    # Get a list of paragraphs based on the inputs
    paras = get_paragraphs(inputs)
    # Get all the relevant context for the paragraphs
    links_dict = get_context_from_paragraphs(paras)
    pdf_file = pdf_creator(links_dict)
    binaryIO = io.BytesIO(pdf_file)
    binaryIO.seek(0)
    return FileResponse(binaryIO, content_type='application/pdf', as_attachment=True, filename='report.pdf')


def create_report(request, inputs, is_authenticated=False):
    # Get a list of paragraphs based on the inputs
    if(request.GET.get("pdfbutton")):
        create_pdf()

    paras = get_paragraphs(inputs)
    report_id = None

    # If a user is logged in then create and save a report instance linked to their profile
    if is_authenticated:
        current_user_profile = models.UserProfile.objects.filter(
            user=request.user).first()
        # Create a report instance and add paragraphs to it
        rep = models.Report(user=current_user_profile,
                            datetime_created=datetime.now())
        rep.save()
        report_id = rep.id
        for p in paras:
            rep.paragraphs.add(p)

    # if a user is not logged in
    else:
        # serialize the paragraphs and save them to the session
        request.session["temp_saved"] = serialize('json', paras)

    return report_id


def pdf_creator(dataset):
    pdf = FPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.add_font('FreeSans', '', './static/fonts/FreeSans.ttf', uni=True)
    pdf.add_font('FreeSans', 'B', './static/fonts/FreeSansBold.ttf', uni=True)
    pdf.add_font('FreeSans', 'I',
                 './static/fonts/FreeSansOblique.ttf', uni=True)
    pdf.image('./static/images/logo_small.png', 10, 8, 33)
    pdf.set_font('FreeSans', 'B', 16)
    pdf.cell(80)
    pdf.cell(30, 10, 'Report', 1, 0, 'C')
    pdf.ln(20)
    pdf.set_font('FreeSans', '', 12)
    for key, values in dataset.items():
        pdf.multi_cell(0, 5, key.static_text)
        pdf.ln('0.1')
        for ls in values:
            for item in ls:
                if(hasattr(item, "url")):
                    pdf.multi_cell(0, 5, "• "+item.url)
                if(hasattr(item, "title")):
                    pdf.multi_cell(0, 5, "• "+item.title)
        pdf.ln('0.3')
    pdf_file = pdf.output(dest='S').encode('latin-1')
    return pdf_file


def get_context_from_paragraphs(paras):
    ''' Helper method retrieving all required data from paragraphs
    that is needed for rendering a report.
    '''
    link_list = models.Link.objects.all()
    actions_list = models.Action.objects.all()
    links_dict = {}

    # get the associated links and actions for each paragraph
    for par in paras:
        temp = []
        t = []
        big_l = []
        for link in link_list:
            if par == link.paragraph:
                temp.append(link)
        for action in actions_list:
            if par == action.paragraph:
                t.append(action)
        if temp:
            big_l.append(temp)
        if t:
            big_l.append(t)
        # list of the lists for links and actions added to dictionary
        links_dict[par] = big_l
    return links_dict


@login_required(login_url='login')
def create_paragraphs(request):
    LinksFormSet = formset_factory(NewLinkForm, extra=3)
    ActionFormSet = formset_factory(NewActionForm, extra=3)
    KeywordFormSet = formset_factory(NewKeywordForm, extra=5)
    if request.method == "POST":
        newParaForm = NewParaForm(request.POST)
        linkformset = LinksFormSet(request.POST)
        actionformset = ActionFormSet(request.POST)
        keywordformset = KeywordFormSet(request.POST)
        if newParaForm.is_valid():
            para = newParaForm.save(commit=False)
            para.created_by = models.UserProfile.objects.filter(
                user=request.user).first()
            para.save()
            if linkformset.is_valid():
                for a_link in linkformset:
                    link = a_link.save(commit=False)
                    if link.url != '':
                        link.paragraph = models.Paragraph.objects.all().order_by('-id').first()
                        link.save()
            if actionformset.is_valid():
                for an_action in actionformset:
                    action = an_action.save(commit=False)
                    if action.title != '':
                        action.paragraph = models.Paragraph.objects.all().order_by('-id').first()
                        action.save()
            if keywordformset.is_valid():
                for a_keyword in keywordformset:
                    keyword = a_keyword.save(commit=False)
                    if keyword.key != '':
                        keyword.paragraph = models.Paragraph.objects.all().order_by('-id').first()
                        keyword.save()
            return redirect(reverse('paragraphs'))
        return redirect(reverse('paragraphs'))
    newParaForm = NewParaForm()
    linkformset = LinksFormSet()
    actionformset = ActionFormSet()
    keywordformset = KeywordFormSet()
    context = {"newParaForm": newParaForm, 'linkformset': linkformset,
               'actionformset': actionformset, 'keywordformset': keywordformset}
    return render(request, 'create_paragraphs.html', context)
