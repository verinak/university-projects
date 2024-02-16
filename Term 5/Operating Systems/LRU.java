import java.util.ArrayList;

// class for page
class Page {
    int pageNo;
    int recency = 0;
    
    public Page(int pageNo){
        this.pageNo = pageNo;
    }
    
    // method to compare pages based on page no
    public boolean equals(Object o){
        if(o instanceof Page){
            return (this.pageNo == ((Page)o).pageNo);
        }
        else {
            return false;
        }
    }
}

class LRU {

    // no of frames in RAM
    final static int NO_OF_FRAMES = 4;
    // the reference string of requests, implemented as an array of integers
    static int[] requests = {7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2};
    
    static int pageFaultCount = 0;
    static ArrayList<Page> RAM = new ArrayList<Page>();
    static ArrayList<Page> zRAM = new ArrayList<Page>();
    
    public static void main(String[] args) {
        
        // iterate over the requests array
        for(int r : requests) {
            // Make a page request
            pageRequest(r);
            System.out.println();
        }
        
        // print the final page fault count
        System.out.println("\nFinal page fault count: " + pageFaultCount);
        
    }
    
    // method to make a page request
    public static void pageRequest(int pageNo) {
        System.out.print("Page Request for Page " + pageNo + "\n");
        
        // increment recency of all pages in RAM
        for (int i = 0; i < RAM.size(); i++) {
            RAM.get(i).recency++;
        }
        
        // if page in ram set recency to 0
        if(RAM.contains(new Page(pageNo))) {
            System.out.print("Page " + pageNo + " is already in RAM\n");
            
            // set recency to 0
            int idx = RAM.indexOf(new Page(pageNo));
            RAM.get(idx).recency = 0;
        }
        // if page not in ram, swap with LRU page
        else {
            System.out.print("Page Fault\n");
            pageFaultCount++;
            swapPage(pageNo);
        }
        
        printRAM();
        
        
    }
    
    // methid to swap requested page with LRU page
    public static void swapPage(int pageNo) {
        
        // if page in zRAM, get it, else create new page object.
        int idx = zRAM.indexOf(new Page(pageNo));
        
        Page requestedPage;
        if(idx == -1) {
            requestedPage = new Page(pageNo);
        }
        else {
            requestedPage = zRAM.get(idx);
            zRAM.remove(idx);
        }
        
        // if RAM is not full, add page. else swap with LRU page
        if(RAM.size() < NO_OF_FRAMES) {
            RAM.add(requestedPage);
        }
        else {
            int swapidx = getLRU();
            Page removedPage = RAM.get(swapidx);
            removedPage.recency = 0;
            zRAM.add(removedPage);
            RAM.set(swapidx, requestedPage);
            System.out.print("Swapped with Page " + removedPage.pageNo + "\n");
            printzRAM();
        }
        
        
    }
    
    // method to find page with highest recency
    public static int getLRU() {
        int idx = 0;
        int max_recency = 0;
        for (int i = 0; i < RAM.size(); i++) {
            int recency = RAM.get(i).recency;
            if(recency > max_recency){
                max_recency = recency;
                idx = i;
            }
        }
        return idx;
        
    }
    
    public static void printRAM() {
        System.out.print("RAM:\n");
        for (int i = 0; i < RAM.size(); i++) {
            Page p = RAM.get(i);
            System.out.print("[" + p.pageNo + ", " + p.recency + "]\n");
        } 
    }
    
    public static void printzRAM() {
        System.out.print("zRAM: (");
        for (int i = 0; i < zRAM.size(); i++) {
            Page p = zRAM.get(i);
            System.out.print(p.pageNo + ", ");
        } 
        System.out.print(")\n");
    }

}
