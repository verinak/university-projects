package hw4;

import java.util.Scanner;

public class SoundBook extends Publication {
	
	private float playingTime;
	
	public void readData() {
		super.readData();
		Scanner console = new Scanner(System.in);
		System.out.print("Playing time (in minutes): ");
		playingTime = console.nextInt();
	}
	
	public void printData() {
		super.printData();
		System.out.println("Playing time: " + playingTime + " minutes");
	}

}
