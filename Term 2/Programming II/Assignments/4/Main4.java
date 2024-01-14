package hw4;

public class Main4 {

	public static void main(String[] args) {
		
		HardBook hb1 = new HardBook();
		SoundBook sb1 = new SoundBook();
		
		System.out.println("Enter hard book data");
		hb1.readData();
		System.out.println("Enter sound book data");
		sb1.readData();
		
		System.out.println();

		System.out.println("Hard book data entered:");
		hb1.printData();
		System.out.println("Hard book data entered:");
		sb1.printData();
		
	}

}
