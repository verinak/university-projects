package hw4;

import java.util.Scanner;

public class Publication {
	
	private String title;
	private float price;
	
	public void readData() {
		Scanner console = new Scanner(System.in);
		System.out.print("Title: ");
		title = console.nextLine();
		System.out.print("Price: ");
		price = console.nextFloat();
	}
	
	public void printData() {
		System.out.println("Title: " + title);
		System.out.println("Price: " + price + "L.E.");
		
	}

}
