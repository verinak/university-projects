import java.util.Scanner;

public class Date {
	
	private int month, day, year;
	
	//constructor setting attributes to 0
	public Date() {
		month = day = year = 0;
	}
	
	//constructor to take date as input
	public Date(String s) {
		getDate(s);
	}
	
	//function to read the date from a string in 12/31/02 format
	public void getDate(String date) {
		String[] substr = {date.substring(0,date.indexOf("/")),
				date.substring(date.indexOf("/")+1, date.lastIndexOf("/")),
				date.substring(date.lastIndexOf("/")+1, date.length())	};
		Scanner scan = new Scanner(substr[0]);
		month = scan.nextInt();
		scan = new Scanner(substr[1]);
		day = scan.nextInt();
		scan = new Scanner(substr[2]);
		year = scan.nextInt();
	}
	
	//accessor and mutator methods
	public int getMonth() {
		return month;
	}
	
	public int getDay() {
		return day;
	}
	
	public int getYear() {
		return year;
	}
	
	public void setMonth(int month) {
		this.month = month;
	}
	
	public void setDay(int day) {
		this.day = day;
	}
	
	public void setYear(int year) {
		this.year = year;
	}
	
	//method to print out the date in 12/31/02 format
	public void showDate() {
		if(month < 10) {
			System.out.print("0" + month + "/");
		}
		else {
			System.out.print(month + "/");
		}
		if(day < 10) {
			System.out.print("0" + day + "/");
		}
		else {
			System.out.print(day + "/");
		}
		if(year < 10) {
			System.out.print("0" + year);
		}
		else {
			System.out.print(year);
		}
		System.out.println();
	}
	
	
}
