package hw7;

import java.util.InputMismatchException;
import java.util.Scanner;

public class sumMain {

	public static void main(String[] args) {
		
		boolean continueOutput = true;
		int n1 = 0, n2 = 0;
		do {
			try {
				Scanner console = new Scanner(System.in);
				System.out.println("Enter two integers");
				n1 = console.nextInt();
				n2 = console.nextInt();
				continueOutput = false;
			}
			catch(InputMismatchException e) {
				System.out.println("Invalid Input. Pease enter integers only.");
			}
		}
		while(continueOutput);
		
		int sum = n1 + n2;
		System.out.println("The sum is " + sum);
		
	}

}
