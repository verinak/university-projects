package queue;

public class Main {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Server comp1 = new Server("Computer 1");
		Server comp2 = new Server("Computer 2");
		
		Server.s.enqueue(new Task());
		Server.s.printList();
		
		new Thread (comp1).start();
		new Thread (comp2).start();
	}

}
