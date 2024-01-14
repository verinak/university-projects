//assignment 1
import java.util.Scanner;
public class Rectangle {
	
	private double length, height;
	
	public Rectangle() {	//contructor
		length = height = 0;
	}
	
	public void setLength() {	//allow user to set length
		Scanner console = new Scanner(System.in);
		System.out.print("Enter length: ");
		length = console.nextDouble();
	}
	
	public void setHeight() {	//allow user to set height
		Scanner console = new Scanner(System.in);
		System.out.print("Enter height: ");
		height = console.nextDouble();
	}
	
	public void displayAttributes() {	//print length and height for user
		System.out.println("Length is " + length);
		System.out.println("Height is " + height);
	}
	
	public void perimeter() {	//calculate perimeter and print it for user
		double perimeter = 2*(length+height);
		System.out.println("The perimeter is " + perimeter);
	}
	
	public void area() {	//calculate area and print it for user
		double area = length*height;
		System.out.println("The area is " + area);
	}
}

//Main
//
/*
 * Rectangle rectangle = new Rectangle();
		rectangle.displayAttributes();
		rectangle.setLength();
		rectangle.setHeight();
		rectangle.displayAttributes();
		rectangle.perimeter();
		rectangle.area();
		*/
