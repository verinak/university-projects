
public class MainTime {

	public static void main(String[] args) {
		
		Time t1 = new Time(11,60,60);
		Time t2 = new Time(12,40,2);
		Time t3 = new Time();
		t3.addTime(t1, t2);
		t3.displayTime();
		
		
		
	}

}
