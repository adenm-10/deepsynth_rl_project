// ******************************************** Iteration:0

#include<stdio.h>
#include<stdbool.h>
#include<stdint.h>
void main()
{
    uint8_t event_seq_length = 2;
   uint8_t num_input = 1;
   uint8_t event_seq[1][2] = {{1,2}};
   uint8_t num_states = 5;
   uint8_t t[3][3];
   uint8_t count=0;
   uint8_t t_gen[1][3] = {{1,1,2}};

   for(uint8_t i=0;i<1;i++)             
        {                                                                       
            t[count][0] = t_gen[i][0];                                          
            t[count][1] = t_gen[i][1];                                          
            t[count][2] = t_gen[i][2];                                  
            count = count + 1;                                                  
        }
                                                                       
    for (uint8_t i=0;i<num_input;i++)                                                  
    {                                                                               
        uint8_t start_state_var;                                                        
        __CPROVER_assume(start_state_var <= num_states && start_state_var > 1);     
        //assert(start_state_var <= num_states);                                        

        t[count][0] = start_state_var;                                              
        for (uint8_t j=0;j<event_seq_length;j++)                                       
        {                                                                           
            uint8_t next_state_var;                                                     
            t[count][1] = event_seq[i][j];                                          
            if(event_seq[i][j] == 1)                                                
                t[count][0] = 1;                                                    
            __CPROVER_assume(next_state_var <= num_states && next_state_var > 1);   
            //assert(next_state_var <= num_states);                                 
            t[count][2] = next_state_var;                                           
            count = count+1;                                                            
            t[count][0] = t[count-1][2];                                            
        }                                                                           
    }

   bool in[5][2] = {false};                                               
    bool o[5][2] = {false};                                                        
                                                                                    
    for (uint8_t i=0;i<count;i++)                                                    
    {                                                                               
        o[t[i][0]-1][t[i][1]-1] = true;                                         
        in[t[i][2]-1][t[i][1]-1] = true;                                            
    }
                                                                       
    bool wrong_transition = false;                                                      
    for (uint8_t i=0; i<num_states;i++)                                                     
    {                                                                               
       if (in[i][ 1] && (o[i][1]))
               wrong_transition = true;
   }
   assert(wrong_transition != false);
}