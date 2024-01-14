
public class Time {
	
	private int hours, minutes, seconds;
	
	//constructor to set all values to 0
	public Time() {
		hours = minutes = seconds = 0;
	}
	
	//constructor to takes values of attributes at initialization 
	public Time(int hours, int minutes, int seconds) {
		this.hours = hours;
		this.minutes = minutes;
		this.seconds = seconds;
		
		checkSecondsMax();
		checkMinutesMax();
	}
	
	//methods to get attributes
	public int getHours() {
		return hours;
	}
	
	public int getMinutes() {
		return minutes;
	}
	
	public int getSeconds() {
		return seconds;
	}
	
	//methods to set attributes
	public void setHours(int hours) {
		this.hours = hours;
	}
	
	public void setMinutes(int minutes) {
		this.minutes = minutes;
	}
	
	public void setSeconds(int seconds) {
		this.seconds = seconds;
	}
	
	//displaying time in 11:59:59 format
	public void displayTime() {
		if(hours < 10) {
			System.out.print("0" + hours + ":");
		}
		else {
			System.out.print(hours + ":");
		}
		if(minutes < 10) {
			System.out.print("0" + minutes + ":");
		}
		else {
			System.out.print(minutes + ":");
		}
		if(seconds < 10) {
			System.out.print("0" + seconds);
		}
		else {
			System.out.print(seconds);
		}
		System.out.println();
	}
	
	//method to add time
	public void addTime(Time time1, Time time2) {
		hours = time1.getHours() + time2.getHours();
		minutes = time1.getMinutes() + time2.getMinutes();
		seconds = time1.getSeconds() + time2.getSeconds();
		
		checkSecondsMax();
		checkMinutesMax();
	}
	
	//methods to make sure minutes and seconds are smaller than 60
	private void checkMinutesMax() {
		while(minutes >= 60) {
			minutes -= 60;
			hours++;
		}
	}
	
	private void checkSecondsMax() {
		while(seconds >= 60) {
			seconds -= 60;
			minutes++;
		}
	}
	

}
