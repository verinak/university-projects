#takes page_frames (the number of page frames) and reference_string (the input reference string) as input
def second_chance(page_frames, reference_string):

    page_faults = 0 # keep track of the number of page faults
    page_table = [] # list is initialized to store the page frames and their reference bits
    clock_hand = 0 # the index of the clock hand in the page frames

    # Initialize page table
    for i in range(page_frames):
        # initialized with each page frame having a reference bit set to 1
        page_table.append({'id':-1, 'bit': 0})  
    # Process reference string
    references = reference_string.split()
    references = [int(r) for r in references]
    print(references)
    print()

    
        
    
    # Iterate over the remaining references
    for reference in references:
        print(f'Page Request for "{reference}"')

        # Look for page with matching id in page_table
        idx = -1
        for page in page_table:
            if page['id'] == reference:
                idx = page_table.index(page)
                break
        
        #If the reference is already present in the page table (page hit), the reference bit of that page frame is set to 1
        if idx != -1:
            
            page_table[idx]['bit'] = 1
            print("Reference found")
            print(clock_hand)
        
        #If the reference is not present in the page table (page fault), the code enters a loop to find a suitable page to replace
        else:
            page_faults += 1
            print("Reference not found.. Adding to memory")
            
            while True:
                print(f'clock hand at frame {clock_hand}')
                print(page_table)

                if page_table[clock_hand]['bit'] == 0:
                    # If the reference bit of the current page frame pointed by the clock hand is 0, it means that 
                    # the page can be replaced. The page table entry for that frame is replaces with the requested reference 
                    # The clock hand is incremented to the next index
                    page_table[clock_hand] = {'id':reference, 'bit': 1}
                    clock_hand = (clock_hand + 1) % page_frames
                    print(clock_hand)
                    break
                else:
                    # Give the page a second chance by setting its bit to 0
                    page_table[clock_hand]['bit'] = 0
                    clock_hand = (clock_hand + 1) % page_frames
                    print(clock_hand)
                    
        print(page_table)
        print()

    # Format page table for output
    page_table_output = [f'{page["id"]}->{page["bit"]}' for page in page_table]
    output = f'Page fault = {page_faults}\n[{", ".join(page_table_output)}]'

    return output


# User input
page_frames = int(input("Enter the number of page frames: "))
reference_string = input("Enter the reference string (separate numbers by spaces): ")

# page_frames = 4
# # reference_string = "1 2 3 4 1 2 5 1 2 3 4 5"
# # reference_string = "1 2 6 5 7 8 9 1 2 3 4 5 6"
# reference_string = "7 0 1 2 0 3 0 4 2 3 0 3 2 3"

print()
# Run the algorithm
output = second_chance(page_frames, reference_string)
print()
print(output)