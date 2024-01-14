package hw8;

import java.io.*;
import java.util.Random;

public class BinaryMain {

	public static void main(String[] args)  throws IOException{
		// TODO Auto-generated method stub
		
		Random rand = new Random();
		int arr[] = new int[100];
		for(int i = 0; i < 100; i++) {
			arr[i] = rand.nextInt(10);
		}
			
		ObjectOutputStream output = new ObjectOutputStream(new FileOutputStream("C:/Users/M/Desktop/Assignment_8.dat"));
		output.writeObject(arr);
		
	}

}
