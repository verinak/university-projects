from flask import Flask, flash, render_template, request, redirect, session
import mysql.connector
from werkzeug.security import check_password_hash, generate_password_hash

app = Flask(__name__)
app.secret_key = 'secret'

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="FacebookProject"
)

@app.route('/')
def home():
    print("home")
    return render_template('home.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':

        firstname = request.form['firstname']
        lastname= request.form['lastname']
        email= request.form['email']
        password = request.form['password']
        phone= request.form['phone']
        date= request.form['birthday']
        about= request.form['about']
        hash=generate_password_hash(password)
        
        cursor = db.cursor()
        query = "INSERT INTO Users (Fname ,Lname , Email, UPassword ,Phone ,Birthday ,About ) VALUES (%s, %s, %s,%s, %s, %s,%s)"
        values = (firstname, lastname,email, hash,phone,date,about)
        cursor.execute(query, values)
        db.commit()

        query = "SELECT UserId FROM Users WHERE Email = %s AND UPassword = %s"
        values = (email,hash)
        cursor.execute(query, values)
        user_id= cursor.fetchone()
        cursor.close()
        print(user_id)

        session['user_id'] = user_id[0]
        return redirect('/dashboard')

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        cursor = db.cursor()
        query = "SELECT * FROM Users WHERE Email=%s"
        values = (email,)
        cursor.execute(query, values)
        user = cursor.fetchone()
        cursor.close()

        print(user)

        if user and check_password_hash(user[4], password):
            session['user_id'] = user[0]
            return redirect('/dashboard')
        else:
            return render_template('login.html', error='Invalid email or password')

    return render_template('login.html')


@app.route('/dashboard')
def dashboard():
    if 'user_id' in session:
        user_id = session['user_id']

        print("Fetching posts...")
        cursor = db.cursor()
        query= """
                (SELECT CONCAT(u.Fname ," ", u.Lname) AS name, u.UserId, up.PostId ,up.PTime,
                p.Title ,p.Likes, up.IsCreated
                FROM users u
                JOIN userpost up on u.UserId= up.UserId
                JOIN post p on p.PostId=up.PostId
                JOIN friends f on f.User1Id = u.UserId
                WHERE f.User2Id = %s)
                UNION
                (SELECT CONCAT(u.Fname ," ", u.Lname) AS name, u.UserId, up.PostId ,up.PTime,
                p.Title ,p.Likes, up.IsCreated
                FROM users u
                JOIN userpost up on u.UserId= up.UserId
                JOIN post p on p.PostId=up.PostId
                JOIN friends f on f.User2Id = u.UserId
                WHERE f.User1Id = %s)
                UNION
                (SELECT pa.Name,pa.PageId,po.PostId,pp.PTime,po.Title,pa.Likes,pp.IsCreated
                FROM page pa
                JOIN pagepost pp on pa.PageId = pp.PageId
                JOIN post po on po.PostId=pp.PostId
                JOIN likespage l on l.PageId = pa.PageId
                WHERE l.UserId= %s)
                ORDER BY PTime DESC
        """

        values = [session['user_id'],session['user_id'],session['user_id']]
        cursor.execute(query, values)
        posts = cursor.fetchall()
        cursor.close()

        return render_template('dashboard.html', posts=posts, user_id=user_id)
    else:
        return redirect('/login')
    
    
@app.route('/share', methods=['GET', 'POST'])
def share():
    print("start share")
    if request.method == 'POST':
        title = request.form['post']
        user_id= session['user_id']


        cursor = db.cursor()
        query = "INSERT INTO Post(Title) VALUES( %s )"
        values = [title]
        cursor.execute(query, values)
        db.commit()
        cursor.close()
        
        cursor = db.cursor()
        query = "INSERT INTO UserPost(UserId,IsCreated,PostId) VALUES (%s,1,(select PostId from post where Title = %s))"
        values = [session['user_id'],title]
        cursor.execute(query, values)
        db.commit()
        cursor.close()

        return redirect('/dashboard')
    return render_template('dashboard.html')

@app.route('/sharefrom<postid>')
def sharefrom(postid):
    print('sharing from friends..')
    postid = postid[1:-1]
    print(postid)

    cursor = db.cursor()
    query = "INSERT INTO UserPost(PostId, UserId, IsCreated) VALUES( %s,%s, 0 )"
    values = [postid,session['user_id']]
    cursor.execute(query, values)
    db.commit()
    cursor.close()
    return redirect('/dashboard')


    
@app.route('/profile')
def profile():
    if 'user_id' in session:
        user_id = session['user_id']
        cursor = db.cursor()
        query = "SELECT * FROM users WHERE UserId=%s"
        values = (user_id,)
        cursor.execute(query, values)
        user = cursor.fetchone()
        cursor.close()

        print("Fetching posts...")
        cursor = db.cursor()
        query = '''SELECT CONCAT(u.Fname ," ", u.Lname) AS name, u.UserId, up.PostId ,up.PTime,
                p.Title ,p.Likes, up.IsCreated
                FROM users u
                JOIN userpost up on u.UserId= up.UserId
                JOIN post p on p.PostId=up.PostId
                WHERE u.UserId = %s
                ORDER BY up.PTime DESC'''

        values = [session['user_id']]
        cursor.execute(query, values)
        posts = cursor.fetchall()
        cursor.close()

        return render_template('profile.html', posts=posts, user=user)
    else:
        return redirect('/login')
    #print("profile done")
    #return render_template('profile.html')
@app.route('/profile/edit', methods=['GET', 'POST'])
def edit_profile():
    if 'user_id' in session:
        user_id = session['user_id']
        cursor = db.cursor()
        query = "SELECT * FROM users WHERE UserId=%s"
        values = (user_id,)
        cursor.execute(query, values)
        user = cursor.fetchone()

        if request.method == 'POST':
            phone = request.form['phone']
            about = request.form['about']
            query = "UPDATE users SET Phone=%s, About=%s WHERE UserId=%s"
            values = (phone, about, user_id)
            cursor.execute(query, values)
            db.commit()
            cursor.close()
            flash("Profile updated successfully.")
            return redirect('/profile')
            #return redirect('/login')

        cursor.close()
        return render_template('edit_profile.html', user=user)
    else:
        return redirect('/login')
    

@app.route('/friends')
def friends():
    if 'user_id' in session:
        user_id = session['user_id']

        cursor = db.cursor()
        query = '''(SELECT CONCAT(u.Fname, " ", u.Lname), u.About
                    FROM friends f 
                    JOIN users u ON f.User2Id = u.UserId
                    WHERE f.User1Id = %s)
                    UNION
                    (SELECT CONCAT(u.Fname, " ", u.Lname), u.About
                    FROM friends f
                    JOIN users u
                    ON f.User1Id = u.UserId
                    WHERE f.User2Id = %s)
                    '''
        values = (session['user_id'],session['user_id'])
        cursor.execute(query, values)
        friends = cursor.fetchall()

        query = '''SELECT CONCAT(Fname, " ", Lname), About, UserId
                    FROM users u
                    WHERE UserId != %s AND UserId NOT IN ((SELECT u.UserId
                    FROM friends f 
                    JOIN users u ON f.User2Id = u.UserId
                    WHERE f.User1Id = %s)
                    UNION
                    (SELECT u.UserId
                    FROM friends f
                    JOIN users u
                    ON f.User1Id = u.UserId
                    WHERE f.User2Id = %s));
                '''
        values = (session['user_id'],session['user_id'],session['user_id'])
        cursor.execute(query, values)
        others=cursor.fetchall()

        cursor.close()

        
        return render_template('friends.html',friends=friends,others=others)
    else:
        return redirect('/login')
    

@app.route('/addfriend<fid>')
def addfriend(fid):

    cursor = db.cursor()
    query = "INSERT INTO Friends(User1Id,User2Id) VALUES( %s,%s)"
    values = [session['user_id'],fid[1:-1]]
    cursor.execute(query, values)
    db.commit()
    cursor.close()

    return redirect('/friends')



@app.route('/pages')
def pages():
    c = db.cursor()
    c.execute('''SELECT Name,CompanyEmail,DateCreated,Likes,PageId FROM page WHERE CreatorId = %s''', (session['user_id'],))
    created_pages = c.fetchall()

    c.execute('''SELECT page.Name,page.CompanyEmail,page.DateCreated,page.Likes,page.PageId FROM page 
                JOIN likespage ON likespage.PageId = page.PageId
                WHERE CreatorId != %s AND likespage.UserId = %s''', (session['user_id'],session['user_id']))
    liked_pages = c.fetchall()

    c.execute('''SELECT page.Name,page.CompanyEmail,page.DateCreated,page.Likes,page.PageId FROM page 
                WHERE CreatorId != %s AND page.PageId NOT IN(SELECT page.PageId FROM page 
                JOIN likespage ON likespage.PageId = page.PageId WHERE likespage.UserId = %s)''', 
                (session['user_id'],session['user_id']))
    other_pages = c.fetchall()

    c.close()

    # Render the "Pages" screen using the retrieved data
    return render_template('pages.html', created=created_pages,liked=liked_pages,other=other_pages)

@app.route('/createpage', methods = ['GET','POST'])
def createPage():
    if request.method == "POST":
        name = request.form['page-name']
        email = request.form['page-email']

        cur = db.cursor()
        query = "INSERT INTO page (Name, CompanyEmail,CreatorId) VALUES (%s,%s,%s)"
        cur.execute(query,(name, email,session['user_id']))

        db.commit()
        cur.close()
        return redirect('/pages')
    else:
        return render_template("createpage.html")

@app.route('/pageprofile<page_id>')
def pageprofile(page_id):

    page_id = page_id[1:-1]

    if 'user_id' in session:
        user_id = session['user_id']

        cursor = db.cursor()
        query = "SELECT Name,Likes FROM page WHERE PageId=%s"
        values = (page_id,)
        cursor.execute(query, values)
        page = cursor.fetchone()
        cursor.close()

        print("Fetching posts...")
        cursor = db.cursor()
        query = '''(select pa.Name,pa.PageId,po.PostId,pa.DateCreated,po.Title,po.Likes,pp.IsCreated
                    from page pa
                    join pagepost pp on pa.PageId = pp.PageId 
                    join post po on po.PostId=pp.PostId
                    where pa.PageId= %s
                    order by pa.DateCreated desc)'''

        values = [page_id]
        cursor.execute(query, values)
        posts = cursor.fetchall()
        cursor.close()

        return render_template('pageprofile.html',page=page,page_id=page_id,posts=posts)
    else:
        return redirect('/login')


@app.route('/sharepage<page_id>', methods=['GET', 'POST'])
def sharepage(page_id):
    print("start share")
    if request.method == 'POST':
        title = request.form['post']

        cursor = db.cursor()
        query = "INSERT INTO post(Title) VALUES( %s )"
        values = [title]
        cursor.execute(query, values)
        db.commit()
        cursor.close()
        
        cursor = db.cursor()
        query = "INSERT INTO pagepost(PageId, IsCreated,Postid) VALUES (%s , 1,(SELECT PostId FROM post WHERE Title = %s ))"
        values = [page_id[1:-1],title]
        cursor.execute(query, values)
        db.commit()
        cursor.close()

        return redirect(f'/pageprofile<{page_id[1:-1]}>')


@app.route('/likepage<page_id>', methods=['GET', 'POST'])
def likepage(page_id):
    page_id = page_id[1:-1]
    user_id = session['user_id']

    cursor = db.cursor()
    query = "INSERT INTO likespage(PageId, UserId) VALUES (%s ,%s)"
    values = [page_id,user_id]
    cursor.execute(query, values)

    query = "UPDATE page SET Likes = Likes + 1 WHERE PageId = %s"
    values = [page_id]
    cursor.execute(query, values)

    db.commit()
    cursor.close()

    return redirect('/pages')

@app.route('/unlikepage<page_id>', methods=['GET', 'POST'])
def unlikepage(page_id):
    page_id = page_id[1:-1]
    user_id = session['user_id']

    cursor = db.cursor()
    query = "DELETE FROM likespage WHERE PageId=%s AND UserId=%s;"
    values = [page_id,user_id]
    cursor.execute(query, values)

    query = "UPDATE page SET Likes = Likes - 1 WHERE PageId = %s"
    values = [page_id]
    cursor.execute(query, values)

    db.commit()
    cursor.close()

    return redirect('/pages')


@app.route('/inbox')
def inbox():

    cursor = db.cursor()
    query = '''(SELECT CONCAT(u.Fname, " ", u.Lname), u.UserId
                FROM friends f 
                JOIN users u ON f.User2Id = u.UserId
                WHERE f.User1Id = %s)
                UNION
                (SELECT CONCAT(u.Fname, " ", u.Lname), u.UserId
                FROM friends f
                JOIN users u
                ON f.User1Id = u.UserId
                WHERE f.User2Id = %s);'''
    values = (session['user_id'],session['user_id'])
    cursor.execute(query, values)
    friends = cursor.fetchall()
    # friends di keda list gowaha tuple l kol row..
    cursor.close()


    return render_template('inbox.html', friends = friends)


@app.route('/chat<user>', methods=['GET', 'POST'])
def chat(user):

    uid = int(user[1:-1]) # el id byerga3 fih <> m3raf4 lehhh
    
    if request.method == 'POST':
        message = request.form['message']

        cursor = db.cursor()
        query = "INSERT INTO chat(Msg,SenderId,ReceiverId) VALUES(%s,%s,%s);"
        values = (message, session['user_id'],uid)
        cursor.execute(query, values)
        db.commit()
        cursor.close()

        return redirect(f'/chat<{uid}>')
    

    cursor = db.cursor()
    query = '''SELECT Msg, CTime, SenderId
                FROM chat
                WHERE (SenderId = %s AND ReceiverId = %s) OR (SenderId = %s AND ReceiverId = %s)
                ORDER BY CTime;'''
    values = (session['user_id'],uid, uid,session['user_id'])
    cursor.execute(query, values)
    chat = cursor.fetchall()
    cursor.close()


    return render_template('chat.html', chat = chat, myid = session['user_id'])


@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)
