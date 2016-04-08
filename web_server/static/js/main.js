
$(function(){
	$('#buttonTest').click(function(){
		var user_id = $('#txtUserId').val();
		var item_id = $('#txtItemId').val();
		var rate_data = '[[5.0, 4.0, -1.0, 3.0],'
		rate_data += '[1.0, 2.0, 3.0, 2.0],'
		rate_data += '[5.0, 4.0, 4.0, 3.0],'
		rate_data += '[4.0, 4.0, 5.0, 3.0],'
		rate_data += '[2.0, 3.0, 4.0, 5.0]]'
		var post_data = {'user_id': user_id, 'item_id': item_id, 'rate_data': rate_data}
		var url_ = 'http://caothanhha9.pythonanywhere.com'
//		var url_ = ''
        $('#txtResult').html('<p>' + 'calculating...' + '</p>')
		$.ajax({
			url: url_ + '/predict',
			data: post_data,
//			headers: {"Access-Control-Allow-Origin": "*"},
			type: 'POST',
			success: function(response){
				console.log(response);
				$('#txtResult').html('<p>' + response + '</p>')
			},
			error: function(error){
				console.log(error);
			}
		});
    });
});
