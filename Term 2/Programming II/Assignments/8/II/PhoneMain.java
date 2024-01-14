package hw8;

import java.io.*;
import java.util.Scanner;

public class PhoneMain {

	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		
		File myFile = new File("C:/Users/M/Desktop/phonebook.txt");
		PrintWriter writer = new PrintWriter(myFile);
		Scanner reader = new Scanner(myFile);
		
		int indexNum = 0;
		
		//add numbers
		writer.println(indexNum + "_ 01234568910");
		indexNum++;
		writer.println(indexNum + "_ 01111111111");
		indexNum++;
		writer.println(indexNum + "_ 0122xxxxxxx");
		indexNum++;
		writer.println(indexNum + "_ 01212121212");
		indexNum++;
		writer.close();
		
		//retrieve numbers
		while(reader.hasNext()) {
			if(reader.hasNext("2_")) {
				break;
			}
			reader.next();
		}
		reader.next();
		System.out.println(reader.next());
		
	}
		
}
