package queue;

import java.util.*;

public class Server implements Runnable {
	
	private String compName;	//34an asamihom comp1 w comp2 34an lma agi atba3 bs ab2a 3arfa anhi task 4a8al 3ala anhi computer
	protected static LinkedListQueue s = new LinkedListQueue();

	public Server(String compName) {	//constructor bya5od esm el computer bs
		this.compName = compName;
	}	

	@Override
	public void run() {		//3ashan elrunnable
			while(true) {
				if(!s.isEmpty()) {	//lw el queue m4 fadya haya5od mnha task w yestana el service time bta3o ba3den yetba3 eno 5alas el task
					Task lastTask = s.dequeue();
					System.out.println("Task " + lastTask.getIndex() + " in progress on " + compName);
					Server.s.printList();
					try {
						Thread.sleep(lastTask.getServiceTime());
					} catch (InterruptedException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
					System.out.println("Task " + lastTask.getIndex() + " done.");
				}
				else {	//lw el queue fadya haystana el average interarrival time (20 secs) w yerga3 tani mn awel el while loop y-check el queue fadya wla la2
					try {
						Thread.sleep(20000);
					} catch (InterruptedException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				}
			}
		
	}
	
}
