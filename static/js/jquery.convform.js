function SingleConvState(input){
    this.input = input;
    this.answer = '';
    this.next = false;
    return this;
};
SingleConvState.prototype.hasNext = function(){
    return this.next;
};
function ConvState(wrapper, SingleConvState, form, params, originalFormHtml) {
    this.form = form;
    this.wrapper = wrapper;
    this.current = SingleConvState;
    this.answers = {};
    this.parameters = params;
    this.originalFormHtml = originalFormHtml;
    this.scrollDown = function() {
        $(this.wrapper).find('#messages').stop().animate({scrollTop: $(this.wrapper).find('#messages')[0].scrollHeight}, 600);
    }.bind(this);
};
