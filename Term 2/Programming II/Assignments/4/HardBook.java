package hw4;

import java.util.Scanner;

public class HardBook extends Publication {
	
	private int pageCount;
	
	public void readData() {
		super.readData();
		Scanner console = new Scanner(System.in);
		System.out.print("Page count: ");
		pageCount = console.nextInt();
	}
	
	public void printData() {
		super.printData();
		System.out.println("Page count: " + pageCount + " pages");
	}

}
