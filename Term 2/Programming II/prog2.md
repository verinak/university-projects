# Programming 2

## Assignments - OOP Lecture Exercises

<details>

<summary>Click to expand</summary>

### Assignment 1
Create a rectangle class with the following:
- The member data should be the length and height of the rectangle.
- Methods to:
    - Allow the user to enter class attributes
    - A method to display the values of the rectangle attributes.
    - A method for calculating the perimeter.
    - A method for calculating the area.
    - A default constructor to initialize the attributes to zero values.

### Assignment 2

I. Create a class called time that has:
- separate int member data for hours, minutes, and seconds.
- One constructor should initialize this data to 0,
- Another constructor initialize it to fixed values.
- Accessor & mutators methods for the attributes.
- A method to display time in 11:59:59 format.
- A method to add two objects of type time passed as arguments.
- In the main() should create two initialized time objects and one that isn't initialized. Then it should add the two initialized values together, leaving the result in the third time variable. Finally, it should display the value of this third variable.


II. Create an employee class.
- The member data should comprise an int for storing the employee number and a float for storing the employee's salary.
- Methods to allow the user to enter this data and display it.
- Apply constructors overloading.
- Accessor & mutators methods for the attributes.


III. Create a date class.
- Its member data should consist of three ints: month, day, and year.
- Two member functions: getDate(), which allows the user to enter a date in 12/31/02 format, and show Date(), which displays the date.
- Apply constructors overloading.
- Accessor & mutators methods for the attributes.

### Assignment 3

I. Design a Shopping Cart program. In this task you will complete a class that implements a shopping cart as an array of items. The Item class models an item one would purchase. An item has a name, price, and quantity (the quantity purchased). The file ShoppingCart.java implements the shopping cart as an array of Item objects. Complete the ShoppingCart class by doing the following:
- Declare an instance variable cart to be an array of Items and instantiate cart in the constructor to be an array holding capacity Items.
- There should be addToCart method. This method should add the item to the cart and update
the totalPrice instance variable (note this variable takes into account the quantity).


II. Design a library program. In this task you will complete a class that
implements a Library as an array of Books. The Book class that models a book one would find the library. A book has a title, price, year, and Author. The Library.java implements the library as an array of Books objects. Complete the Library class by doing the following:
- Declare an instance variable library to be an array of Books and instantiate library in the constructor to be an array holding capacity Items.
- There should be addBook method. This method should add the book to the library.

### Assignment 4

1. Imagine a publishing company that markets both printed book and soft versions of its works. Create a class publication that stores the title (a string) and price (type float) of a publication. From this class derive two classes: book, which adds a page count (type int), and soundBook, which adds a playing time in minutes (type float). Each of these three classes should have a readData() method to get its data from the user at the keyboard, and a printData() method to display its data.
2. Write a main() program to test the hardBook and soundBook classes by creating instances of them, asking the user to fill in data with readData(), and then displaying the data with printData().

### Assignment 5

- The Account class is defined to model a bank account. An account class has:
    - Add a field name of the String type to store the name of the customer.
    - Other attributes include account number, balance, annual interest rate, and date created,
    - A no-arg constructor that creates a default account.
    - A constructor that constructs an account with the specified name, id, and balance.
    - Add a data field named transactions whose type is ArrayList that stores the transaction for the accounts. Each transaction include.
        - The date of this transaction.
        - The type of the transaction, such as 'W' for withdrawal, 'D' for deposit.
        - The amount of the transaction.
        - The new balance after this transaction.
        - Construct a Transaction with the specified date, type, balance, and description.
    - Methods to deposit and withdraw funds that adds the transaction to the arrayList.
- Create two subclasses for checking and saving accounts.
    - A checking account has an overdraft limit, but a savings account cannot be overdrawn.


1. Draw the UML diagram for the classes and then implement them.
2. Write a test program that creates objects of Account, Savings Account, and CheckingAccount and invokes their toString() methods.
3. Creates an Account with annual interest rate 1.5%, balance 1000, id 1122, and name Sarah .
4. Deposit $30, $40, and $50 to the account and withdraw $5, $4, and $2 from the account.
5. Print an account summary that shows account holder name, interest rate, balance, and all transactions.

### Assignment 6

- Design a Triangle class that extends the abstract GeometricObject class. Draw the UML diagram for the classes Triangle and GeometricObject and then implement the Triangle class.
- Write a test program that prompts the user to enter three sides of the triangle, a color, and a Boolean value to indicate whether the triangle is filled.
    - The program should create a Triangle object with these sides and set the color and filled properties using the input.
    - The program should display the area, perimeter, color, and true or false to indicate whether it is filled or not.
    - Make the class comparable on the basis of the area.

### Assignment 7

I. Write a program that prompts the user to read two integers and displays their sum. Your program should prompt the user to read the number again if the input is incorrect. (using InputMismatchException)


II. Write a program that meets the following requirements:
- Creates an array with 100 randomly chosen integers.
- Prompts the user to enter the index of the array, then displays the corresponding element value. If the specified index is out of bounds, display the message Out of Bounds. (using ArrayIndexOutOfBoundsException)


III. Modify the Triangle class, in Assignment 6, with three sides. In a triangle, the sum of any two sides is greater than the other side. The Triangle class must adhere to this rule.
- Create the IllegalTriangleException class, and modify the constructor of the Triangle class to throw an Illegal Triangle Exception object if a triangle is created with sides that violate the rule:


``` java
/** Construct a triangle with the specified sides */
public Triangle(double sidel, double side2, double side3) throws IllegalTriangleException {
    // Implement it
}
```

### Assignment 8

I. (Create a binary data file) Write a program to create a file named Assignment_8.dat if it does not exist.
- Append new data to it if it already exists.
- Write 100 integers created randomly into the file using binary I/O.


II. Write a program that stores, retrieves, adds, and updates Phone numbers.

</details>

## Course Project

Update the Bank Account from Assignment 5 to use File I/O to store user data.


## After Course

I wanted to create something beyond concole apps and Java was the only programming language I could use at that point, so I played around with Java Swing for a while during summer break.
The most impressive thing I managed was a Puzzle game. <!-- ba3den rabena hadani w ro7t at3alem python ta2riban-->


### Project Features

- Puzzle Pieces can be dragged and dropped.
- Pieces are locked on the board once they are in the right place.
- When all pieces are in place, "Puzzle Complete" window is shown.
- Preview of final image can be shown or hidden.
- Reload Puzzle button to reset the puzzle.

<img src="After Course/ss.png" height=400px></img>

