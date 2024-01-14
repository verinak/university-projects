package hw7;

import java.util.Scanner;

public class arrayMain {

	public static void main(String[] args) {
		
		int[] arr = new int[100];
		for(int i = 0; i < 100; i++) {
			arr[i] = (int)(Math.random()*100);
		}
		
		try {
			Scanner console = new Scanner(System.in);
			System.out.println("Pick an Index");
			System.out.println(arr[console.nextInt()]);
		}
		catch(ArrayIndexOutOfBoundsException e) {
			System.out.println("Out of Bounds");
		}
	}

}
