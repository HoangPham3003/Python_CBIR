from flask import session, redirect, request, abort, render_template


class SystemController:
    @staticmethod
    def login_required(func):
        def secure_function(**kwargs):
            if 'auth' not in session:
                return redirect('/signin')
            return func(**kwargs)
        return secure_function

    @staticmethod
    def check_acl(func):
        def check_acl_function(**kwargs):
            request_path = request.path
            if len(kwargs):
                # last position is always kwargs
                index = request_path.rfind('/')
                action_check =  request_path.replace('/', '.')[1:index] # have page index or other kwargs
            elif '?' in request_path:
                index = request_path.index('?')
                action_check = request_path.replace('/', '.')[1:index]
            else:
                action_check = request_path.replace('/', '.')[1:]
            public_action = ['', 'signin', 'register', 'forgot-password', 'signout', 'page', 'images.category', 'about-us']
            if action_check in public_action:
                return func(**kwargs)
            if 'auth' in session:
                info_user = session['auth']
                if action_check not in info_user['Pms_Name']:
                    return render_template('exceptionview/403.html')
                return func(**kwargs)
            else:
                return render_template('exceptionview/404.html') 
        return check_acl_function

