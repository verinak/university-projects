package hw6;

import java.util.Scanner;

public class MainGeometric {

	public static void main(String[] args) {
		Scanner console = new Scanner(System.in);
		System.out.print("Enter the 3 sides of the trangle: ");
		double s1 = console.nextDouble(); 
		double s2 = console.nextDouble(); 
		double s3 = console.nextDouble(); 
		System.out.print("Enter the color of the triangle (one word only): ");
		String color = console.next();
		System.out.print("Is the triangle filled? (true/false): ");
		boolean filled = console.nextBoolean();
		console.close();
		
		System.out.println();
		
		Triangle t1 = new Triangle(s1,s2,s3,color,filled);
		System.out.println(t1.toString());
		System.out.println("Area: " + t1.getArea());
		System.out.println("Perimeter: " + t1.getPerimeter());

		Triangle t2 = new Triangle(10,12,20);
		Triangle t3 = new Triangle(1,1,1);
		Triangle t4 = new Triangle(3,4,5);
		
		System.out.println("t1.compareTo(t2): " + t1.compareTo(t2));
		System.out.println("t1.compareTo(t3): " + t1.compareTo(t3));
		System.out.println("t1.compareTo(t4): " + t1.compareTo(t4));
		
	}
	

}
