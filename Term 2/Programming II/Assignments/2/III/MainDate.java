import java.util.Scanner;

public class MainDate {

	public static void main(String[] args) {
		
		Scanner console = new Scanner(System.in);
		System.out.println("Enter a date (mm/dd/yy): ");
		String date = console.next();
		Date today = new Date(date);
		today.showDate();
		today.setDay(18);
		System.out.println("Month: " + today.getMonth());
		System.out.println("Day: " + today.getDay());
		System.out.println("Year: " + today.getYear());

		
	}

}
