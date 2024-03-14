from unicodedata import category
from aiohttp import request
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.urls import reverse
from django.contrib.auth import authenticate, login, logout, update_session_auth_hash
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
import json
from datetime import datetime,time
from django.contrib.auth.models import User
from django.contrib import messages
from django.db.models import Q
from ams.settings import MEDIA_ROOT, MEDIA_URL
from django import forms
from attendance.models import Attendance, UserProfile,Course, Department,Student, Class, ClassStudent
from attendance.forms import UserRegistration, UpdateProfile, UpdateProfileMeta, UpdateProfileAvatar, AddAvatar, SaveDepartment, SaveCourse, SaveClass, SaveStudent, SaveClassStudent, UpdatePasswords, UpdateFaculty

from attendance.forms import StudentcodeForm
from django.shortcuts import render,redirect
from .forms import DateForm,UsernameAndDateForm, DateForm_2,VurlForm
from django.contrib import messages
from django.contrib.auth.models import User
import cv2
import dlib
import imutils
from imutils import face_utils
from imutils.video import VideoStream
from imutils.face_utils import rect_to_bb
from imutils.face_utils import FaceAligner
import time
from ams.settings import BASE_DIR
import os
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
from django.contrib.auth.decorators import login_required
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import datetime,time
from django_pandas.io import read_frame
# from users.models import Present, Time
import seaborn as sns
import pandas as pd
from django.db.models import Count
#import mpld3
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from matplotlib import rcParams
import math

# VURL='http://192.168.137.16:8080/video'


deparment_list = Department.objects.exclude(status = 2).all()
context = {
    'page_title' : 'Simple Blog Site',
    'deparment_list' : deparment_list,
    'deparment_list_limited' : deparment_list[:3]
}
#login
def login_user(request):
    logout(request)
    resp = {"status":'failed','msg':''}
    username = ''
    password = ''
    if request.POST:
        username = request.POST['username']
        password = request.POST['password']

        user = authenticate(username=username, password=password)
        if user is not None:
            if user.is_active:
                login(request, user)
                resp['status']='success'
            else:
                resp['msg'] = "Incorrect username or password"
        else:
            resp['msg'] = "Incorrect username or password"
    return HttpResponse(json.dumps(resp),content_type='application/json')

#Logout
def logoutuser(request):
    logout(request)
    return redirect('/')

@login_required
def home(request):
    context['page_title'] = 'Home'
    departments = Department.objects.count()
    courses = Course.objects.count()
    faculty = UserProfile.objects.filter(user_type = 2).count()
    if request.user.profile.user_type == 1:
        students = Student.objects.count()
        classes = Class.objects.count()
    else:
        classes = Class.objects.filter(assigned_faculty = request.user.profile).count()
        students = ClassStudent.objects.filter(classIns__in = Class.objects.filter(assigned_faculty = request.user.profile).values_list('id')).count()
    context['departments'] = departments
    context['courses'] = courses
    context['faculty'] = faculty
    context['students'] = students
    context['classes'] = classes

    # context['posts'] = posts
    return render(request, 'home.html',context)

def registerUser(request):
    user = request.user
    if user.is_authenticated:
        return redirect('home-page')
    context['page_title'] = "Register User"
    if request.method == 'POST':
        data = request.POST
        form = UserRegistration(data)
        if form.is_valid():
            form.save()
            newUser = User.objects.all().last()
            try:
                profile = UserProfile.objects.get(user = newUser)
            except:
                profile = None
            if profile is None:
                UserProfile(user = newUser, dob= data['dob'], contact= data['contact'], address= data['address'], avatar = request.FILES['avatar']).save()
            else:
                UserProfile.objects.filter(id = profile.id).update(user = newUser, dob= data['dob'], contact= data['contact'], address= data['address'])
                avatar = AddAvatar(request.POST,request.FILES, instance = profile)
                if avatar.is_valid():
                    avatar.save()
            username = form.cleaned_data.get('username')
            pwd = form.cleaned_data.get('password1')
            loginUser = authenticate(username= username, password = pwd)
            login(request, loginUser)
            return redirect('home-page')
        else:
            context['reg_form'] = form

    return render(request,'register.html',context)

@login_required
def profile(request):
    context = {
        'page_title':"My Profile"
    }

    return render(request,'profile.html',context)
    
@login_required
def update_profile(request):
    context['page_title'] = "Update Profile"
    user = User.objects.get(id= request.user.id)
    profile = UserProfile.objects.get(user= user)
    context['userData'] = user
    context['userProfile'] = profile
    if request.method == 'POST':
        data = request.POST
        # if data['password1'] == '':
        # data['password1'] = '123'
        form = UpdateProfile(data, instance=user)
        if form.is_valid():
            form.save()
            form2 = UpdateProfileMeta(data, instance=profile)
            if form2.is_valid():
                form2.save()
                messages.success(request,"Your Profile has been updated successfully")
                return redirect("profile")
            else:
                # form = UpdateProfile(instance=user)
                context['form2'] = form2
        else:
            context['form1'] = form
            form = UpdateProfile(instance=request.user)
    return render(request,'update_profile.html',context)


@login_required
def update_avatar(request):
    context['page_title'] = "Update Avatar"
    user = User.objects.get(id= request.user.id)
    context['userData'] = user
    context['userProfile'] = user.profile
    if user.profile.avatar:
        img = user.profile.avatar.url
    else:
        img = MEDIA_URL+"/default/default-avatar.png"

    context['img'] = img
    if request.method == 'POST':
        form = UpdateProfileAvatar(request.POST, request.FILES,instance=user)
        if form.is_valid():
            form.save()
            messages.success(request,"Your Profile has been updated successfully")
            return redirect("profile")
        else:
            context['form'] = form
            form = UpdateProfileAvatar(instance=user)
    return render(request,'update_avatar.html',context)

@login_required
def update_password(request):
    context['page_title'] = "Update Password"
    if request.method == 'POST':
        form = UpdatePasswords(user = request.user, data= request.POST)
        if form.is_valid():
            form.save()
            messages.success(request,"Your Account Password has been updated successfully")
            update_session_auth_hash(request, form.user)
            return redirect("profile")
        else:
            context['form'] = form
    else:
        form = UpdatePasswords(request.POST)
        context['form'] = form
    return render(request,'update_password.html',context)

#Department
@login_required
def department(request):
    departments = Department.objects.all()
    context['page_title'] = "Department Management"
    context['departments'] = departments
    return render(request, 'department_mgt.html',context)

@login_required
def manage_department(request,pk=None):
    # department = department.objects.all()
    if pk == None:
        department = {}
    elif pk > 0:
        department = Department.objects.filter(id=pk).first()
    else:
        department = {}
    context['page_title'] = "Manage Department"
    context['department'] = department

    return render(request, 'manage_department.html',context)

@login_required
def save_department(request):
    resp = { 'status':'failed' , 'msg' : '' }
    if request.method == 'POST':
        department = None
        print(not request.POST['id'] == '')
        if not request.POST['id'] == '':
            department = Department.objects.filter(id=request.POST['id']).first()
        if not department == None:
            form = SaveDepartment(request.POST,instance = department)
        else:
            form = SaveDepartment(request.POST)
    if form.is_valid():
        form.save()
        resp['status'] = 'success'
        messages.success(request, 'Department has been saved successfully')
    else:
        for field in form:
            for error in field.errors:
                resp['msg'] += str(error + '<br>')
        if not department == None:
            form = SaveDepartment(instance = department)
       
    return HttpResponse(json.dumps(resp),content_type="application/json")

@login_required
def delete_department(request):
    resp={'status' : 'failed', 'msg':''}
    if request.method == 'POST':
        id = request.POST['id']
        try:
            department = Department.objects.filter(id = id).first()
            department.delete()
            resp['status'] = 'success'
            messages.success(request,'Department has been deleted successfully.')
        except Exception as e:
            raise print(e)
    return HttpResponse(json.dumps(resp),content_type="application/json")


#Course
@login_required
def course(request):
    courses = Course.objects.all()
    context['page_title'] = "Course Management"
    context['courses'] = courses
    return render(request, 'course_mgt.html',context)

@login_required
def manage_course(request,pk=None):
    # course = course.objects.all()
    if pk == None:
        course = {}
        department = Department.objects.filter(status=1).all()
    elif pk > 0:
        course = Course.objects.filter(id=pk).first()
        department = Department.objects.filter(Q(status=1) or Q(id = course.id)).all()
    else:
        department = Department.objects.filter(status=1).all()
        course = {}
    context['page_title'] = "Manage Course"
    context['departments'] = department
    context['course'] = course

    return render(request, 'manage_course.html',context)

@login_required
def save_course(request):
    resp = { 'status':'failed' , 'msg' : '' }
    if request.method == 'POST':
        course = None
        print(not request.POST['id'] == '')
        if not request.POST['id'] == '':
            course = Course.objects.filter(id=request.POST['id']).first()
        if not course == None:
            form = SaveCourse(request.POST,instance = course)
        else:
            form = SaveCourse(request.POST)
    if form.is_valid():
        form.save()
        resp['status'] = 'success'
        messages.success(request, 'Course has been saved successfully')
    else:
        for field in form:
            for error in field.errors:
                resp['msg'] += str(error + '<br>')
        if not course == None:
            form = SaveCourse(instance = course)
       
    return HttpResponse(json.dumps(resp),content_type="application/json")

@login_required
def delete_course(request):
    resp={'status' : 'failed', 'msg':''}
    if request.method == 'POST':
        id = request.POST['id']
        try:
            course = Course.objects.filter(id = id).first()
            course.delete()
            resp['status'] = 'success'
            messages.success(request,'Course has been deleted successfully.')
        except Exception as e:
            raise print(e)
    return HttpResponse(json.dumps(resp),content_type="application/json")

#Faculty
@login_required
def faculty(request):
    user = UserProfile.objects.filter(user_type = 2).all()
    context['page_title'] = "Faculty Management"
    context['faculties'] = user
    return render(request, 'faculty_mgt.html',context)

@login_required
def manage_faculty(request,pk=None):
    if pk == None:
        faculty = {}
        department = Department.objects.filter(status=1).all()
    elif pk > 0:
        faculty = UserProfile.objects.filter(id=pk).first()
        department = Department.objects.filter(Q(status=1) or Q(id = faculty.id)).all()
    else:
        department = Department.objects.filter(status=1).all()
        faculty = {}
    context['page_title'] = "Manage Faculty"
    context['departments'] = department
    context['faculty'] = faculty
    return render(request, 'manage_faculty.html',context)

@login_required
def view_faculty(request,pk=None):
    if pk == None:
        faculty = {}
    elif pk > 0:
        faculty = UserProfile.objects.filter(id=pk).first()
    else:
        faculty = {}
    context['page_title'] = "Manage Faculty"
    context['faculty'] = faculty
    return render(request, 'faculty_details.html',context)

@login_required
def save_faculty(request):
    resp = { 'status' : 'failed', 'msg' : '' }
    if request.method == 'POST':
        data = request.POST
        if data['id'].isnumeric() and data['id'] != '':
            user = User.objects.get(id = data['id'])
        else:
            user = None
        if not user == None:
            form = UpdateFaculty(data = data, user = user, instance = user)
        else:
            form = UserRegistration(data)
        if form.is_valid():
            form.save()

            if user == None:
                user = User.objects.all().last()
            try:
                profile = UserProfile.objects.get(user = user)
            except:
                profile = None
            if profile is None:
                form2 = UpdateProfileMeta(request.POST,request.FILES)
            else:
                form2 = UpdateProfileMeta(request.POST,request.FILES, instance = profile)
                if form2.is_valid():
                    form2.save()
                    resp['status'] = 'success'
                    messages.success(request,'Faculty has been save successfully.')
                else:
                    User.objects.filter(id=user.id).delete()
                    for field in form2:
                        for error in field.errors:
                            resp['msg'] += str(error + '<br>')
            
        else:
            for field in form:
                for error in field.errors:
                    resp['msg'] += str(error + '<br>')

    return HttpResponse(json.dumps(resp),content_type='application/json')

@login_required
def delete_faculty(request):
    resp={'status' : 'failed', 'msg':''}
    if request.method == 'POST':
        id = request.POST['id']
        try:
            faculty = User.objects.filter(id = id).first()
            faculty.delete()
            resp['status'] = 'success'
            messages.success(request,'Faculty has been deleted successfully.')
        except Exception as e:
            raise print(e)
    return HttpResponse(json.dumps(resp),content_type="application/json")


    
#Class
@login_required
def classPage(request):
    if request.user.profile.user_type == 1:
        classes = Class.objects.all()
    else:
        classes = Class.objects.filter(assigned_faculty = request.user.profile).all()

    context['page_title'] = "Class Management"
    context['classes'] = classes
    return render(request, 'class_mgt.html',context)

@login_required
def manage_class(request,pk=None):
    faculty = UserProfile.objects.filter(user_type= 2).all()
    if pk == None:
        _class = {}
    elif pk > 0:
        _class = Class.objects.filter(id=pk).first()
    else:
        _class = {}
    context['page_title'] = "Manage Class"
    context['faculties'] = faculty
    context['class'] = _class

    return render(request, 'manage_class.html',context)

@login_required
def view_class(request, pk= None):
    if pk is None:
        return redirect('home-page')
    else:
        _class = Class.objects.filter(id=pk).first()
        students = ClassStudent.objects.filter(classIns = _class).all()
        context['class'] = _class
        context['students'] = students
        context['page_title'] = "Class Information"
    return render(request, 'class_details.html',context)


@login_required
def save_class(request):
    resp = { 'status':'failed' , 'msg' : '' }
    if request.method == 'POST':
        _class = None
        print(not request.POST['id'] == '')
        if not request.POST['id'] == '':
            _class = Class.objects.filter(id=request.POST['id']).first()
        if not _class == None:
            form = SaveClass(request.POST,instance = _class)
        else:
            form = SaveClass(request.POST)
    if form.is_valid():
        form.save()
        resp['status'] = 'success'
        messages.success(request, 'Class has been saved successfully')
    else:
        for field in form:
            for error in field.errors:
                resp['msg'] += str(error + '<br>')
        if not _class == None:
            form = SaveClass(instance = _class)
       
    return HttpResponse(json.dumps(resp),content_type="application/json")

@login_required
def delete_class(request):
    resp={'status' : 'failed', 'msg':''}
    if request.method == 'POST':
        id = request.POST['id']
        try:
            _class = Class.objects.filter(id = id).first()
            _class.delete()
            resp['status'] = 'success'
            messages.success(request,'Class has been deleted successfully.')
        except Exception as e:
            raise print(e)
    return HttpResponse(json.dumps(resp),content_type="application/json")

@login_required
def manage_class_student(request,classPK = None):
    if classPK is None:
        return HttpResponse('Class ID is Unknown')
    else:
        context['classPK'] = classPK
        _class  = Class.objects.get(id = classPK)
        # print(ClassStudent.objects.filter(classIns = _class))
        students = Student.objects.exclude(id__in = ClassStudent.objects.filter(classIns = _class).values_list('student').distinct()).all()
        context['students'] = students
        return render(request, 'manage_class_student.html',context)
@login_required
def save_class_student(request):
    resp = {'status' : 'failed', 'msg':''}
    if request.method == 'POST':
        form = SaveClassStudent(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request,"Student has been added successfully.")
            resp['status'] = 'success'
        else:
            for field in form:
                for error in field.errors:
                    resp['msg'] += str(error+"<br>")
    return HttpResponse(json.dumps(resp),content_type = 'json')

@login_required
def delete_class_student(request):
    resp={'status' : 'failed', 'msg':''}
    if request.method == 'POST':
        id = request.POST['id']
        try:
            cs = ClassStudent.objects.filter(id = id).first()
            cs.delete()
            resp['status'] = 'success'
            messages.success(request,'Student has been deleted from Class successfully.')
        except Exception as e:
            raise print(e)
    return HttpResponse(json.dumps(resp),content_type="application/json")


#Student
@login_required
def student(request):
    students = Student.objects.all()
    context['page_title'] = "Student Management"
    context['students'] = students
    return render(request, 'student_mgt.html',context)

@login_required
def manage_student(request,pk=None):
    # course = course.objects.all()
    if pk == None:
        student = {}
        course = Course.objects.filter(status=1).all()
    elif pk > 0:
        student = Student.objects.filter(id=pk).first()
        course = Course.objects.filter(Q(status=1) or Q(id = course.id)).all()
    else:
        course = Course.objects.filter(status=1).all()
        student = {}
    context['page_title'] = "Manage Course"
    context['courses'] = course
    context['student'] = student

    return render(request, 'manage_student.html',context)
@login_required
def view_student(request,pk=None):
    if pk == None:
        student = {}
    elif pk > 0:
        student = Student.objects.filter(id=pk).first()
    else:
        student = {}
    context['student'] = student
    return render(request, 'student_details.html',context)

@login_required
def save_student(request):
    resp = { 'status':'failed' , 'msg' : '' }
    if request.method == 'POST':
        student = None
        print(not request.POST['id'] == '')
        if not request.POST['id'] == '':
            student = Student.objects.filter(id=request.POST['id']).first()
        if not student == None:
            form = SaveStudent(request.POST,instance = student)
        else:
            form = SaveStudent(request.POST)
    if form.is_valid():
        form.save()
        resp['status'] = 'success'
        messages.success(request, 'Student Details has been saved successfully')
    else:
        for field in form:
            for error in field.errors:
                resp['msg'] += str(error + '<br>')
        if not course == None:
            form = SaveStudent(instance = course)
       
    return HttpResponse(json.dumps(resp),content_type="application/json")

@login_required
def delete_student(request):
    resp={'status' : 'failed', 'msg':''}
    if request.method == 'POST':
        id = request.POST['id']
        try:
            student = Student.objects.filter(id = id).first()
            student.delete()
            resp['status'] = 'success'
            messages.success(request,'Student Details has been deleted successfully.')
        except Exception as e:
            raise print(e)
    return HttpResponse(json.dumps(resp),content_type="application/json")

#Attendance
@login_required
def attendance_class(request):
    if request.user.profile.user_type == 1:
        classes = Class.objects.all()
    else:
        classes = Class.objects.filter(assigned_faculty = request.user.profile).all()
    context['page_title'] = "Attendance Management"
    context['classes'] = classes
    return render(request, 'attendance_class.html',context)


@login_required
def attendance(request,classPK = None):
    _class = Class.objects.get(id = classPK)
    students = Student.objects.filter(id__in = ClassStudent.objects.filter(classIns = _class).values_list('student')).all()
    context['page_title'] = "Attendance Management"
    context['class'] = _class
    if request.method=='POST':
        form=DateForm(request.POST)
        data = request.POST.copy()
        date1=data.POST('date')
        print(date1)
    else:
        today=datetime.date.today()
        print( today)
    attendance = Attendance.objects.filter(attendance_date=today,classIns = _class).all()
    context['attendance']=attendance
    context['form']  = DateForm
    return render(request, 'attendance_mgt.html',context)



 # recognition
@login_required
def add_photos(request):

	if request.method=='POST':
		form=StudentcodeForm(request.POST)
		data = request.POST.copy()
		studentcode=data.get('studentcode')
        
		if Student.objects.filter(student_code=studentcode).exists():
			create_dataset(studentcode)
			messages.success(request, f'Dataset Created')
			return redirect('add-photos')
		else:
			messages.warning(request, f'No such student code found. Please register student first.')
			return redirect('home-page')


	else:
		

			form=StudentcodeForm()
			return render(request,'add_photos.html', {'form' : form})

def create_dataset(username):
	id = username
	if(os.path.exists('face_recognition_data/training_dataset/{}/'.format(id))==False):
		os.makedirs('face_recognition_data/training_dataset/{}/'.format(id))
	directory='face_recognition_data/training_dataset/{}/'.format(id)

	# Detect face
	#Loading the HOG face detector and the shape predictpr for allignment

	print("[INFO] Loading the facial detector")
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')   #Add path to the shape predictor ######CHANGE TO RELATIVE PATH LATER
	fa = FaceAligner(predictor , desiredFaceWidth = 96)
	#capture images from the webcam and process and detect the face
	# Initialize the video stream
	print("[INFO] Initializing Video stream")
	# vs = VideoStream(VURL).start()
	vs = VideoStream(src=0).start()
	#time.sleep(2.0) ####CHECK######

	# Our identifier
	# We will put the id here and we will store the id with a face, so that later we can identify whose face it is
	
	# Our dataset naming counter
	sampleNum = 0
	# Capturing the faces one by one and detect the faces and showing it on the window
	while(True):
		# Capturing the image
		#vs.read each frame
		frame = vs.read()
		#Resize each image
		frame = imutils.resize(frame ,width = 800)
		#the returned img is a colored image but for the classifier to work we need a greyscale image
		#to convert
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		#To store the faces
		#This will detect all the images in the current frame, and it will return the coordinates of the faces
		#Takes in image and some other parameter for accurate result
		faces = detector(gray_frame,0)
		#In above 'faces' variable there can be multiple faces so we have to get each and every face and draw a rectangle around it.
		
		
			


		for face in faces:
			print("inside for loop")
			(x,y,w,h) = face_utils.rect_to_bb(face)

			face_aligned = fa.align(frame,gray_frame,face)
			# Whenever the program captures the face, we will write that is a folder
			# Before capturing the face, we need to tell the script whose face it is
			# For that we will need an identifier, here we call it id
			# So now we captured a face, we need to write it in a file
			sampleNum = sampleNum+1
			# Saving the image dataset, but only the face part, cropping the rest
			
			if face is None:
				print("face is none")
				continue


			

			cv2.imwrite(directory+'/'+str(sampleNum)+'.jpg'	, face_aligned)
			face_aligned = imutils.resize(face_aligned ,width = 400)
			#cv2.imshow("Image Captured",face_aligned)
			# @params the initial point of the rectangle will be x,y and
			# @params end point will be x+width and y+height
			# @params along with color of the rectangle
			# @params thickness of the rectangle
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
			# Before continuing to the next loop, I want to give it a little pause
			# waitKey of 100 millisecond
			cv2.waitKey(50)

		#Showing the image in another window
		#Creates a window with window name "Face" and with the image img
		cv2.imshow("Add Images",frame)
		#Before closing it we need to give a wait command, otherwise the open cv wont work
		# @params with the millisecond of delay 1
		cv2.waitKey(1)
		#To get out of the loop
		if(sampleNum>50):
			break
	
	#Stoping the videostream
	vs.stop()
	# destroying all the windows
	cv2.destroyAllWindows()


#training the model

def train(request):
	
	training_dir='face_recognition_data/training_dataset'
	
	
	
	count=0
	for person_name in os.listdir(training_dir):
		curr_directory=os.path.join(training_dir,person_name)
		if not os.path.isdir(curr_directory):
			continue
		for imagefile in image_files_in_folder(curr_directory):
			count+=1

	X=[]
	y=[]
	i=0


	for person_name in os.listdir(training_dir):
		print(str(person_name))
		curr_directory=os.path.join(training_dir,person_name)
		if not os.path.isdir(curr_directory):
			continue
		for imagefile in image_files_in_folder(curr_directory):
			print(str(imagefile))
			image=cv2.imread(imagefile)
			try:
				X.append((face_recognition.face_encodings(image)[0]).tolist())
				

				
				y.append(person_name)
				i+=1
			except:
				print("removed")
				os.remove(imagefile)

			


	targets=np.array(y)
	encoder = LabelEncoder()
	encoder.fit(y)
	y=encoder.transform(y)
	X1=np.array(X)
	print("shape: "+ str(X1.shape))
	np.save('face_recognition_data/classes.npy', encoder.classes_)
	svc = SVC(kernel='linear',probability=True)
	svc.fit(X1,y)
	svc_save_path="face_recognition_data/svc.sav"
	with open(svc_save_path, 'wb') as f:
		pickle.dump(svc,f)

	
	vizualize_Data(X1,targets)
	
	messages.success(request, f'Training Complete.')

	return render(request,"train.html")

def vizualize_Data(embedded, targets,):
	
	X_embedded = TSNE(n_components=2).fit_transform(embedded)

	for i, t in enumerate(set(targets)):
		idx = targets == t
		plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=t)

	plt.legend(bbox_to_anchor=(1, 1));
	rcParams.update({'figure.autolayout': True})
	plt.tight_layout()	
	plt.savefig('./static/plot/training_visualisation.png')
	plt.close()

def mark_your_attendance(request):
	
	

	
	detector = dlib.get_frontal_face_detector()
	
	predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')   #Add path to the shape predictor ######CHANGE TO RELATIVE PATH LATER
	svc_save_path="face_recognition_data/svc.sav"	


		
			
	with open(svc_save_path, 'rb') as f:
			svc = pickle.load(f)
	fa = FaceAligner(predictor , desiredFaceWidth = 96)
	encoder=LabelEncoder()
	encoder.classes_ = np.load('face_recognition_data/classes.npy')


	faces_encodings = np.zeros((1,128))
	no_of_faces = len(svc.predict_proba(faces_encodings)[0])
	count = dict()
	present = dict()
	log_time = dict()
	start = dict()
	for i in range(no_of_faces):
		count[encoder.inverse_transform([i])[0]] = 0
		present[encoder.inverse_transform([i])[0]] = False

	

	# vs = VideoStream(VURL).start()
	vs = VideoStream(src=0).start()
	
	sampleNum = 0
	
	while(True):
		
		frame = vs.read()
		
		frame = imutils.resize(frame ,width = 800)
		
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		faces = detector(gray_frame,0)
		
		


		for face in faces:
			print("INFO : inside for loop")
			(x,y,w,h) = face_utils.rect_to_bb(face)

			face_aligned = fa.align(frame,gray_frame,face)
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
					
			
			(pred,prob)=predict(face_aligned,svc)
			

			
			if(pred!=[-1]):
				
				person_name=encoder.inverse_transform(np.ravel([pred]))[0]
				pred=person_name
				if count[pred] == 0:
					start[pred] = time.time()
					count[pred] = count.get(pred,0) + 1

				if count[pred] == 4 and (time.time()-start[pred]) > 1.2:
					 count[pred] = 0
				else:
				#if count[pred] == 4 and (time.time()-start) <= 1.5:
					present[pred] = True
					log_time[pred] = datetime.datetime.now()
					count[pred] = count.get(pred,0) + 1
					print(pred, present[pred], count[pred])
				cv2.putText(frame, str(person_name)+ str(prob), (x+6,y+h-6), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

			else:
				person_name="unknown"
				cv2.putText(frame, str(person_name), (x+6,y+h-6), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

			
			
			#cv2.putText()
			# Before continuing to the next loop, I want to give it a little pause
			# waitKey of 100 millisecond
			#cv2.waitKey(50)

		#Showing the image in another window
		#Creates a window with window name "Face" and with the image img
		cv2.imshow("Mark Attendance - In - Press q to exit",frame)
		#Before closing it we need to give a wait command, otherwise the open cv wont work
		# @params with the millisecond of delay 1
		#cv2.waitKey(1)
		#To get out of the loop
		key=cv2.waitKey(50) & 0xFF
		if(key==ord("q")):
			break
	
	#Stoping the videostream
	vs.stop()

	# destroying all the windows
	cv2.destroyAllWindows()
	update_attendance(present)
	return redirect('home-page')

def predict(face_aligned,svc,threshold=0.7):
	face_encodings=np.zeros((1,128))
	try:
		x_face_locations=face_recognition.face_locations(face_aligned)
		faces_encodings=face_recognition.face_encodings(face_aligned,known_face_locations=x_face_locations)
		if(len(faces_encodings)==0):
			return ([-1],[0])

	except:

		return ([-1],[0])

	prob=svc.predict_proba(faces_encodings)
	result=np.where(prob[0]==np.amax(prob[0]))
	if(prob[0][result[0]]<=threshold):
		return ([-1],prob[0][result[0]])

	return (result[0],prob[0][result[0]])


# def update_attendance(present):
#     today=datetime.date.today()
#     time_now=datetime.datetime.now()
#     current_time = time_now.strftime("%H:%M:%S")

#     for person in present:
#         student=Student.objects.get(student_code=person)
#         classin=ClassStudent.objects.get(student=person)
#         try:
# 		   qs=Attendance.objects.get(student=student,attendance_date=today)
# 		except :
# 			qs= None
		
# 		if qs is None:
# 			if present[person]==True:
#                     if current_time<time(13,00):
# 						a=Attendance(student=student,classIns=classin,attendance_date=today,firstpresent=True,secondpresent=False,total=0.5)
# 						a.save()
#                     else:
#                         a=Attendance(student=student,classIns=classin,attendance_date=today,firstpresent=false,secondpresent=True,total=0.5)
# 						a.save()     
# 			# else:
# 			# 	a=Present(user=user,date=today,present=False)
# 			# 	a.save()
# 		else:
# 			if present[person]==True:
# 				if current_time>time(13,00):
#                     qs.secondpresent=True
#                 qs.total= total(qs.firstpresent,qs.secondpresent)
#             	qs.save(update_fields=['secondpresent','total'])


def update_attendance(present):
	today=datetime.date.today()
	time_now=datetime.datetime.now()
	current_time1= time_now.strftime("%H:%M:%S")

	for person in present:
		student=Student.objects.get(student_code=person)
		classin=ClassStudent.objects.get(student=student)
		print(student)
		print(classin)
		try:
		   qs=Attendance.objects.get(student=student,attendance_date=today)
		except :
			qs= None
		
		if qs is None:
			if present[person]==True:
					if current_time1<"13":
						a=Attendance(student=student,classIns=classin.classIns,attendance_date=today,firstpresent=True,secondpresent=False,total=0.5)
						a.save()
					else:
						a=Attendance(student=student,classIns=classin.classIns,attendance_date=today,firstpresent=False,secondpresent=True,total=0.5)
						a.save()	 
			# else:
			# 	a=Present(user=user,date=today,present=False)
			# 	a.save()
		else:
			if present[person]==True:
				if current_time1>"13":
				    qs.secondpresent=True
				    qs.total= total(qs.firstpresent,qs.secondpresent)
				    qs.save(update_fields=['secondpresent','total'])	
				





def total(firstpresent,secondpresent):

	total=0

	if firstpresent==False and secondpresent==False:
		total=0
	elif firstpresent==True and secondpresent==False:
		total=0.5
	elif firstpresent==False and secondpresent==True:
		total=0.5
	else:
		total=1

	return(total)

def add_vurl(request):

	if request.method=='POST':
		form=VurlForm(request.POST)
		data = request.POST.copy()
		VURL=str(data.get('vurl'))
		print(VURL) 
		return redirect('home-page')   
	else:
		

		form=VurlForm()
		return render(request,'add_vurl.html', {'form' : form})