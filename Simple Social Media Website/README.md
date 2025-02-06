# Introduction to Databases

## Project Requirements

- Design ERD and Schema with entities (Login – User – Page – Post – Friend – [User-Post Relationship] - Chat)
- Implement database in PHPMyAdmin with MYSQL
- Create a login and signup interface and connect it with database
- Create simple page to add new post and view logged in user posts

## Our Work

### ERD:

<img src="media/erd.png" height="500px"></img>

### Relational Schema: 

<img src="media/schema.png" height="300px"></img>

### Tools

We used Flask to create the application, and Xampp to access PHPMyAdmin.<br/>

### Application Overview

The application starts with a welcome page.

<img width="800px" src="media/1.png" alt="welcome page"><br/><br/>

The user can then choose to either Sign Up or Log In.

<img width="800px" src="media/2.png" alt="sign up page"><br/>
<img width="800px" src="media/3.png" alt="login page"><br/><br/>

After the user is registered, they land in this homepage, which consists of a navigation sidebar and the main feed. The main feed is where the user can see their friends posts, and share them if they would like. The user can also write their own posts.

<img width="800px" src="media/4.png" alt="home page"><br/><br/>

The sidebar contains 5 buttons. Each button redirects to a specific page.<br/>
The first button is My Profile. Here, the user can view their profile details and their posts, and edit their profile info. If the user clicks the Edit Profile button, a new page opens up where they can see the editable attributes of their profile and change them.

<img width="800px" src="media/5.png" alt="my profile page"><br/>
<img width="800px" src="media/6.png" alt="edit profile page"><br/>

Here is the effect of updating the About attribute.<br/>
<img width="800px" src="media/7.png" alt="edit profile output"><br/><br/>

The second button is the Pages button. This is where the user can create a page, manage their created pages, and like or unlike other's pages. The Create Page button at the top opens a page that prompts the user to enter the page name and the company email.
The Edit Your Page buttons allows the user to view their page's profile, where they can see the page's posts or create new ones.

<img width="800px" src="media/8.png" alt="pages page"><br/>
<img width="800px" src="media/9.png" alt="create page page"><br/>
<img width="800px" src="media/10.png" alt="page profile page"><br/>

Here is the effect of clicking the Unlike button for a liked page.<br/>
<img width="800px" src="media/11.png" alt="unlike page output"><br/><br/>

The third button is the Friends button. This shows the users their friends, and allows them to add other users as friends.
<img width="800px" src="media/12.png" alt="friends page"><br/><br/>

The fourth button is Inbox. Here the user can see a list of friends they can chat with. On clicking a certain friend's name, they can access the chat with this friend, where they can see previously sent messages, and send new ones.

<img width="800px" src="media/13.png" alt="inbox page"><br/>
<img width="800px" src="media/14.png" alt="chat page"><br/><br/>

The fifth button is Log Out. This button simply logs the user out and redirects them to the welcome page.<br/><br/>

## Team Members

Verina Michel<br/>
Marly Magdy<br/>
Ola Mamdouh<br/>
Maria Anwar<br/>
Mirna Tarek<br/>
Mariem Nasr<br/>
