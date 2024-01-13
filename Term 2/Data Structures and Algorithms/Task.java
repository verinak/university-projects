package queue;

import java.util.*;

public class Task {
	
	private static int taskCount = 0;	//mlha4 lazma di bs 34an a3raf asamihom task1, task2, ... keda
	private int index;
	private long serviceTime, interTime;
	
	public Task() {
		index = ++taskCount;
		serviceTime = generateTime(25000);
		interTime = generateTime(20000);
		
		//hayestana wa2t el interarrival w y-generate task gdid w y7oto fl queue
		Timer timer = new Timer();
		TimerTask t = new TimerTask(){
			public void run() {
				Server.s.enqueue(new Task());
				Server.s.printList();
			}
		};
		timer.schedule(t, interTime);
	}
	
	public String toString() {
		return ("Task " + index);
	}
	
	public int getTaskCount() {
		return taskCount;
	}
	
	public int getIndex() {
		return index;
	}

	public long getServiceTime() {
		return serviceTime;
	}

	public long getInterTime() {
		return interTime;
	}
	
	//method to generate a random time following exponential distribution using a uniform random variable x
	public long generateTime(double mean) {
		double x;
		x=-Math.log(1-Math.random())*mean;
		return (long) x;
	}

}
